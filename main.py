import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision
import seaborn as sns

torch.manual_seed(0)

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

# Split train_data into train and validation sets
val_data = torch.utils.data.Subset(train_data, range(50000, 51000))

# Reduce the size of the training set to 5,000
train_data = torch.utils.data.Subset(train_data, range(0, 5000))

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class LeNet5(nn.Module):
    def __init__(self, kernel_size):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), 2) 
        x = self.conv2(x) 
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(-1, 256) 
        x = self.fc1(x) 
        x = F.relu(x)
        x = self.fc2(x) 
        x = F.relu(x)
        x = self.fc3(x) 
        return x

# Initialize kernel size
kernel_size = st.sidebar.slider('Kernel Size', min_value=1, max_value=5, value=5, step=1)

model = LeNet5(kernel_size=kernel_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs=3

train_losses = []
val_losses = []

for epoch in range(epochs):
    t_loss= 0.0
    v_loss=0.0

    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        t_loss+=loss.item()*data.size(0)

    model.eval()
    for data, target in val_loader:
        output=model(data)
        loss=criterion(output,target)
        v_loss+= loss.item()*data.size(0)

    train_loss= t_loss/len(train_loader.sampler)
    val_loss = v_loss/len(val_loader.sampler)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        val_loss
        ))

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()

# Add slider for kernel size
kernel_size = st.slider('Select kernel size', 3, 7, step=2)

with torch.no_grad():
    correct=0
    total=0
    for data, target in test_loader:
        output=model(data)
        _, y_hat = torch.max(output.data, 1)
        total+=target.size(0)
        correct+=(y_hat==target).sum().item()

    print('Test Accuracy: {}%'.format(100 * correct / total))


test_img = train_data[1][0].unsqueeze(0)
conv1=F.relu(model.conv1(test_img))
c1 = conv1 - conv1.min()
c1 = c1 / conv1.max()

fig,axes=plt.subplots(2,3,figsize=(20,10))
ax=axes.ravel()

for i in range(6):
    sns.heatmap(c1[0][i].detach().numpy(),ax=ax[i],cmap='gray')
    ax[i].set_title('Image {}'.format(i+1))
    ax[i].set_xticks([])
    ax[i].set_yticks([])

# Change kernel size
model.conv1 = nn.Conv2d(1, 6, kernel_size)

st.pyplot(fig)
