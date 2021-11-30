import torch 
import matplotlib.pyplot as plt
import numpy as np
from torch import nn 

######################### CONSTANTS #################################

NUM_EPOCHS = 10
LIN = 784
H1 = 125
H2 = 65
NUM_CLASSES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###################### MODEL DEFINITION #############################

class CNN(nn.Module):

	def __init__(self):
		super().__init__()
		#input channels, output channels, kernel size, stride
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		#input channels, nodes 
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, NUM_CLASSES)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x,2 ,2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*(500))
		x = F.relu(self.fc1())
		#raw output of the network for the crossentropy loss
		#log probabilities using the score (raw ouput)
		x = self.fc2(x)
		return x


model = CNN()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10
running_loss_history = []

for e in range(epochs):
	pass		
