# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms


VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG11(nn.Module):
    def __init__(self, in_channels=1, num_classes=2965):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG11'])
        
        self.fcs = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)


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
model = VGG11().to(device)

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
            