# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, num_classes=2965):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(p=0.25)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.linear1 = nn.Linear(256*2*2, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.dropout(x))
        x = self.relu(self.conv2(x))
        x = self.pool(self.dropout(x))
        x = self.relu(self.conv3(x))
        x = self.pool(self.dropout(x))
        x = self.relu(self.conv4(x))
        x = self.pool(self.dropout(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x


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
model = LeNet().to(device)

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
            