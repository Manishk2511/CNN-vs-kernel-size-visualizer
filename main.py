import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

st.title("CNN vs kernel size")

st.write(
    "This app shows changes in training loss, validation loss, testing accuracy and convolution output with respect to kernel size"
)

st.write("Dataset Used : MNIST")
st.write("MNIST dataset contains images of handwritten digits from 0 to 9. It has 60,000 training images and 10,0000 testing images.")
st.write('---')



train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

val_data = torch.utils.data.Subset(train_data, range(50000, 51000))

train_data = torch.utils.data.Subset(train_data, range(0, 5000))

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class CNN(nn.Module):
    def __init__(self, kernel_size):
        self.kernel_size=kernel_size
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size)
        self.fc1 = nn.Linear(6 * ((28-kernel_size+1)//2)*((28-kernel_size+1)//2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), 2) 
        x = x.view(-1, 6 * ((28-self.kernel_size+1)//2)*((28-self.kernel_size+1)//2))
        x = self.fc1(x) 
        x = F.relu(x)
        x = self.fc2(x) 
        x = F.relu(x)
        x = self.fc3(x) 
        return x

kernel_size = st.sidebar.slider('Kernel Size', min_value=1, max_value=10, value=5, step=1)


model = CNN(kernel_size=kernel_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs= st.sidebar.slider('Number of Epochs',min_value=1,max_value=20,value=5,step=1)

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



st.write(
    "Plot for training loss vs epochs and validation loss vs epochs"
)

fig,ax=plt.subplots()
ax.plot(train_losses, label='Training loss')
ax.plot(val_losses, label='Validation loss')
ax.set_title('Training Loss vs Epochs And Validation loss vs Epochs')
ax.set_xlabel('Epochs')
ax.set_ylabel('Training Loss and Validation Loss')
ax.legend()

st.pyplot(fig)

Test_accuracy=None

with torch.no_grad():
    correct=0
    total=0
    for data, target in test_loader:
        output=model(data)
        _, y_hat = torch.max(output.data, 1)
        total+=target.size(0)
        correct+=(y_hat==target).sum().item()

    Test_accuracy=(100 * correct / total)


st.write('Test Accuracy : ',Test_accuracy)
st.write('Description for the above plot : ')
st.write('As we increase the kernel size, the training and validation loss decreases. A large kernel size helps in learning more complex features, therefore it performs well in classification and give less training and validation loss. ')
st.write('The test accuracy also increases with the increase in kernel size because the model has learned the complex features hence they can more accurately perform classification.')

st.write('---')

fig,ax=plt.subplots()
ax.set_title('Image from train set')
ax.imshow(train_data[5][0].reshape((28,28)),cmap='gray')
st.pyplot(fig)

test_img = train_data[5][0].unsqueeze(0)
conv1=F.relu(model.conv1(test_img))
c1 = conv1 - conv1.min()
c1 = c1 / conv1.max()


st.write('---')

st.write('Plot of different images coming out from convolution layer after applying different filters')
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
ax = axes.ravel()

for i in range(6):
    img = c1[0][i].detach().numpy()
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title('Image {}'.format(i+1))
    ax[i].set_xticks([])
    ax[i].set_yticks([])


st.pyplot(fig)
st.write('Description for the above plot : ')
st.write('The output from the convolution layers suggests that the larger size kernels are able to extract more complex features from the image compared to smaller kernel size. As we can see from the above plot, more complex edges and curves present in the image is captured by the larger size kernel compared to the smaller size kernels. ')
