import torch

import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision import transforms
from srae import RAEClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} to train networks.")


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