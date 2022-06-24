# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=1
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=1
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    

def ResNet18(img_channel=1, num_classes=2965):
    return ResNet(block, [2, 2, 2, 2], img_channel, num_classes)


# Igual ----------------------------------------------------

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x) 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0)
    model.train()
    return float(num_correct)/float(num_samples)*100

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = 'kanji'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.Resize(size=(32, 32)),
                                transforms.RandomAffine(degrees=5,
                                                        translate=(0.02, 0.02),
                                                        scale=(0.95, 1.05))])
dataset = datasets.ImageFolder(root, transform=transform)

# Hyperparameters
num_classes = 2965
learning_rate = 1e-4
weight_decay = 0
momentum = 0
num_epochs = 500
n_prints = 100

# Load Data
train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),
                                                              len(dataset) - int(len(dataset)*0.8)])

train_batch_size = 16
test_batch_size = 16
train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=True)

# Initialize network
model = ResNet18().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda e: 0.1)


best_acc = 0
# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # zero previous gradients
        optimizer.zero_grad()

        # back-propagation
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        #if (batch_idx + 1) % (len(train_loader) // n_prints) == 0:
        #    train_acc = check_accuracy(train_loader, model)
        #    test_acc = check_accuracy(test_loader, model)
        #    print(f'Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx + 1}/{len(train_loader)}: '
        #          f'Train. Acc.: {train_acc:.4f}, Test Acc.: {test_acc:.4f}, LR: {scheduler.get_last_lr()}')
        
    #if epoch >= 20:
    #    if (epoch - 4) % 20 == 0:
    #        scheduler.step()
    print(epoch)
    
    train_acc = check_accuracy(train_loader, model)
    test_acc = check_accuracy(test_loader, model)
    if test_acc > best_acc:
        best_acc = test_acc
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}
        torch.save(state, f'{model.__class__.__name__}.pth')
        with open(f'{model.__class__.__name__}_log.txt', 'w') as f:
            f.write(f'Epoch: {epoch}, Train acc.: {train_acc:.4f}, Test acc.: {test_acc:.4f}')
            