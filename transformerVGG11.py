# Imports
import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
import torchvision
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice and easy way
import torchvision.transforms as transforms # Transformations we can perform on our dataset

import os
import pandas as pd
import math


# Encoder
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
        
        #self.linear = nn.Linear(512, 256 * 4)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
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
    
vgg = VGG11()
state_dict = torch.load('VGG11.pth')['state_dict']

remove_layers = []
for s in state_dict.keys():
    if s.startswith('fcs'):
        remove_layers.append(s)
        
for l in remove_layers:
    del state_dict[l]

vgg.load_state_dict(state_dict)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vgg = vgg
        self.linear = nn.Linear(512, 256 * 4)
        
    def forward(self, x):
        x = self.vgg(x)
        x = self.linear(x)
        x = x.reshape(x.shape[0], 256, -1)
        return x
    
encoder = Encoder()


# Igual ----------------------------------------------------
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
root = 'kanji/'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.Resize(size=(32, 32)),
                                transforms.RandomAffine(degrees=5,
                                                        translate=(0.02, 0.02),
                                                        scale=(0.95, 1.05))])
dataset = datasets.ImageFolder(root, transform=transform)


# Target seq
kanjis = list(dataset.class_to_idx.keys())

with open('ids.txt', 'r', encoding='utf-8') as f:
    ids = f.readlines()
ids = ids[2:]

def clean_string(s):
    lst_s = s.split('\n')[0].split('\t')
    return lst_s[1:]
    
ids_lists = list(map(clean_string, ids))

sequences = []
for kanji in kanjis:
    for lst in ids_lists:
        if kanji == lst[0]:
            sequences.append(lst[1])
            break
            
target_seq = dict(zip(kanjis, sequences))
max_len = max(list(map(len, list(target_seq.values())))) + 1


# Vocabulary
sequence_elements = []
for seq in sequences:
    for char in seq:
        sequence_elements.append(char)
        
vocab_size = len(pd.Categorical(sequence_elements).categories) + 3  # last 3 positions are <start>, <end> and <void> tokens
codes = pd.Categorical(pd.Categorical(sequence_elements).categories).codes

vocab_codes = dict(zip(pd.Categorical(sequence_elements).categories, codes))


# Modify dataset
def seq2codes(seq):
    return list(map(lambda x: vocab_codes[x], seq))

def labels2embed(labels):
    labels = list(labels)
    seqs_list = list(map(lambda l: target_seq[kanjis[l]], labels))
    codes_list = list(map(seq2codes, seqs_list))
    one_hot_list = list(map(lambda code: F.one_hot(torch.tensor(code).long(),
                                                   num_classes=vocab_size),
                            codes_list))
    one_hot_list = list(map(lambda t: torch.cat([F.one_hot(torch.tensor([vocab_size - 3]), num_classes=vocab_size),
                                                 t,
                                                 torch.ones(max_len - t.shape[0] - 1, vocab_size) * (vocab_size - 1)],
                                                dim=0),
                            one_hot_list))
    return torch.stack(one_hot_list, dim=0).transpose(0, 1)

def labels2embed_output(labels):
    labels = list(labels)
    seqs_list = list(map(lambda l: target_seq[kanjis[l]], labels))
    codes_list = list(map(seq2codes, seqs_list))
    one_hot_list = list(map(lambda code: torch.cat([torch.tensor(code + [vocab_size - 2]),
                                                    torch.ones(max_len - len(code) - 1) * (vocab_size - 1)],
                                                   dim=0),
                            codes_list))
    return torch.stack(one_hot_list, dim=0)  # batch x seq


# Loss
def check_loss(loader, model):
    losses = []
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)  # 64 x 1 x 28 x 28
            targets = targets.to(device)  # batch

            # forward propagation
            scores = model(data, targets)
            #print(scores.shape)
            loss = criterion(scores.permute(1, 2, 0),
                             labels2embed_output(targets).to(device).long())
            losses.append(loss)
    model.train()
    return torch.tensor(losses).mean().item()


# Model
class PositionalEncoding(nn.Module):

    def __init__(self, d_model=vocab_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if pe.shape[1] % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1)  # seq x batch x feature
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    
    def __init__(self, encoder=encoder, d_model=256, nhead=8, dim_feedforward=512, num_layers=6):
        super().__init__()
        
        # Encoder
        self.encoder = encoder
        
        # Decoder
        self.linear1 = nn.Linear(vocab_size, 256)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=num_layers)
        self.linear2 = nn.Linear(256, vocab_size)
        #self.softmax = nn.Softmax(2)
        
        # Positional encoding
        self.pe = PositionalEncoding()
        
    def decoder_mask(self, seq_len=max_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    def forward(self, images, labels):
        # Encoder
        enc_images = self.encoder(images)
        enc_images = enc_images.reshape(enc_images.shape[0], 256, -1).permute(2, 0, 1)  # seq x batch x feature
        #print(enc_images.shape)
        
        # Decoder
        embedding = labels2embed(labels).to(device)  # seq x batch x feature
        dec_output = self.decoder(self.linear1(self.pe(embedding)),
                                  enc_images,
                                  self.decoder_mask())
        output = self.linear2(dec_output)
        #print(output.shape)
        
        return output
    
    
# Hyperparameters
lr = 0.1
rho = 0.95
eps = 1e-4

num_epochs = 100
n_prints = 1


# Load Data
train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),
                                                              len(dataset) - int(len(dataset)*0.8)])

train_batch_size = 16  #len(train_set) // 1000
test_batch_size = 16
train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=True)


# Initialize network
model = Transformer().to(device)
model.load_state_dict(torch.load('Transformer_VGG11.pth')['state_dict'])


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=rho, eps=eps)

#best_loss = check_loss(test_loader, model)
#print(best_loss)


# Train Network
best_loss = 1000
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device)  # 64 x 1 x 28 x 28
        targets = targets.to(device)  # batch

        # forward propagation
        scores = model(data, targets)
        #print(scores.shape)
        loss = criterion(scores.permute(1, 2, 0),
                         labels2embed_output(targets).to(device).long())

        # zero previous gradients
        optimizer.zero_grad()

        # back-propagation
        # calculate the gradients in back-propagation and update the weights with the optimizer
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
        
    print(epoch)
    train_loss = check_loss(train_loader, model)
    test_loss = check_loss(test_loader, model)
    print(f'Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx + 1}/{len(train_loader)}: '
          f'Train. Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    if test_loss < best_loss:
        best_loss = test_loss
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
        torch.save(state, f'{model.__class__.__name__}_VGG11.pth')
        with open(f'{model.__class__.__name__}_VGG11_log.txt', 'w') as f:
            f.write(f'Epoch: {epoch}, Train loss.: {train_loss:.6f}, Test loss.: {test_loss:.6f}')
            
