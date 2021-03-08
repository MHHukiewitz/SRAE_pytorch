import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets as datasets

from typing import Callable


# TODO: Merging and resetting features
# TODO: Plots of layer activities
# TODO: Stack RAEs for deep network
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} to train networks.")


class RAEClassifier(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'output_size']

    def __init__(self, input_size, hidden_size, output_size,
                 reconstruction_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 hidden_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 output_activation: Callable[[torch.Tensor], torch.Tensor] = nn.Softmax(),
                 reconstruction_loss: nn.Module = nn.MSELoss()):
        super(RAEClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input = torch.zeros(input_size)

        self.output_activation = output_activation

        # also possible: CosineEmbeddingLoss
        self.reconstruction_loss = reconstruction_loss

        self.autoencoder = ReactiveAutoencoder(input_size, hidden_size, self.reconstruction_loss,
                                               hidden_activation,
                                               reconstruction_activation)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.classifier.weight.register_hook(self.backward_classifier_hook)

    def forward(self, input):
        """The forward pass calculates only the h if no error_signal is provided."""
        self.input = input
        encoding, reconstruction = self.autoencoder(input)  # Forward the Autoencoder and detach from the graph
        output = self.classifier(encoding)  # Forward the detached h through the Classifier

        return self.output_activation(output)

    def backward_classifier_hook(self, grad):
        """Triggers autoencoder sparsification with classifier, after backward on this classifier."""
        with torch.enable_grad():
            encoding, reconstruction = self.autoencoder(self.input, torch.sum(grad, 0))
            rec_loss = self.reconstruction_loss(reconstruction, self.input)
            rec_loss.backward()


class ReactiveAutoencoder(nn.Module):
    """The RAE a.k.a. SRAE a.k.a. the autoencoder with the strict supervised sparsity matrix.
    This module provides a framework for training an encoder to maximize information throughput,
    while converging on an error_signal. Works currently only for single samples/online learning.
    Planned are batch mode as well as multiple layers."""

    __constants__ = ['input_size', 'output_size']

    def __init__(self, input_size, output_size, reconstruction_loss: nn.Module,
                 hidden_activation: Callable[[torch.Tensor], torch.Tensor] = None,
                 reconstruction_activation: Callable[[torch.Tensor], torch.Tensor] = None,
                 bias=True, reconstruction_bias: str = 'zeros', activation_scaling=True):
        super(ReactiveAutoencoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation  # TODO: what happens if both activations differ?
        self.activation_scaling = activation_scaling
        if activation_scaling:
            self.scaling = None  # TODO: Really necessary?

        self.encoder = nn.Linear(input_size, output_size, bias=bias)
        self.h = torch.zeros(output_size, requires_grad=True)
        self.predict = torch.zeros(output_size)

        self.reconstruction_activation = reconstruction_activation
        self.reconstruction_loss = reconstruction_loss
        self.reconstructed_input = torch.zeros(input_size, requires_grad=True)

        self.reconstruction_bias_type = reconstruction_bias
        self.reconstruction_bias = self.fresh_reconstruction_bias(self.reconstruction_bias_type)

    def fresh_reconstruction_bias(self, type: str):
        if type == 'none':
            return None
        elif type == 'zeros':
            return torch.zeros(self.input_size, requires_grad=True).to(device)  # TODO: Automatically cast to device
        elif type == 'ones':
            return torch.ones(self.input_size, requires_grad=True).to(device),
        elif type == 'rand':
            return torch.rand(self.input_size, requires_grad=True).to(device),
        elif type == 'randn':
            return torch.randn(self.input_size, requires_grad=True).to(device),

    def forward(self, x: torch.Tensor, error_signal: torch.Tensor = None):
        """The forward pass calculates only the h if no error_signal is provided.
        If an error_signal is provided, then assume same x and use the last h for sparsity and
        reconstruction calculation.
        """
        # first pass forward
        if error_signal is None:
            with torch.no_grad():
                self.h = self.encoder(x)
                if self.hidden_activation is not None:
                    # save for later
                    self.h = self.hidden_activation(self.h)
            return self.h, None

        # reconstruction
        self.h.requires_grad_()
        self.reconstructed_input = F.linear(self.h, self.encoder.weight.t(), self.reconstruction_bias)
        if self.reconstruction_activation is not None:
            self.reconstructed_input = self.reconstruction_activation(self.reconstructed_input)
        # calculate preliminary loss
        rec_loss = self.reconstruction_loss(self.reconstructed_input, x)
        rec_loss.backward()  # compute gradients for self.encoder.weight & self.bias
        # compute strict supervised sparsity mask
        # predict output after update
        self.predict = F.linear(x, self.encoder.weight + self.encoder.weight.grad,
                                self.encoder.bias)
        delta = self.h - self.predict
        if self.activation_scaling:
            # adjust own gradient scaling to error_signal magnitude -> compare maxima
            self.scaling = (torch.max(torch.abs(error_signal)).item() / torch.max(delta).item())
            adjusted_delta = delta * self.scaling
            # noinspection PyTypeChecker
            mask = torch.where((error_signal - adjusted_delta).abs() < error_signal.abs(), 1, 0)
        else:
            # noinspection PyTypeChecker
            mask = torch.where((error_signal - delta).abs() < error_signal.abs(), 1, 0)
        # reset gradients from preliminary backward calculation
        self.encoder.zero_grad()
        masked_encoding = self.h * mask

        # reconstruct using sparsified h
        self.reconstructed_input = F.linear(masked_encoding, self.encoder.weight.t(), self.reconstruction_bias)

        return self.h, self.reconstructed_input

    def backward(self):
        super(ReactiveAutoencoder, self).backward()
        if self.activation_scaling:
            self.encoder.weight.grad *= self.scaling
            self.encoder.bias.grad *= self.scaling
            self.reconstruction_bias.grad += self.scaling

    def reset_parameters(self) -> None:
        super(ReactiveAutoencoder, self).reset_parameters()
        self.reconstruction_bias = self.fresh_reconstruction_bias(self.reconstruction_bias_type)


def one_hot(index: int, classes: int) -> torch.Tensor:
    y = torch.zeros(classes)
    y[index] = 1
    return y


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))
])

target_transform = transforms.Compose([
    transforms.Lambda(lambda x: one_hot(x, 10))
])

# Prepare MNIST dataset
mnist = datasets.MNIST('./datasets/', train=True, download=True, transform=transform, target_transform=target_transform)
print(mnist)
trainset, testset = data.random_split(mnist, [50000, 10000])
# TODO: DataLoader

# N is batch size; D_in is x dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, mnist[0][0].shape[0], 196, 10
epochs = 25

# Construct our autoencoder by instantiating the class defined above
classifier = RAEClassifier(D_in, H, D_out).to(device)

# Construct our loss function and an Optimizer. The call to autoencoder.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the autoencoder.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
for ep in range(epochs):
    print(f"Epoch {ep}...")
    for i, xy in enumerate(trainset):
        x = xy[0].to(device)
        y = xy[1].to(device)
        # Forward pass: Compute predicted y by passing x to the autoencoder
        y_pred = classifier(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if i % 9999 == 0:
            print(loss.item())
            print("Label:", torch.argmax(y).item(), "Predicted:", torch.argmax(y_pred).item())

        # Zero gradients, perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Validating...")
    hit = 0
    total_loss = 0
    for x, y in testset:
        y_pred = classifier(x.to(device))
        total_loss += criterion(y_pred, y.to(device))
        if torch.argmax(y).item() == torch.argmax(y_pred).item():
            hit += 1

    total_loss /= len(testset)
    hit_rate = hit / len(testset) * 100
    print(f"Validation MSELoss: {total_loss}, accuracy: {hit_rate:.3f}")
