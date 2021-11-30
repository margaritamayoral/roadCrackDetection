import torch 
import torch.nn as nn
import torch.nn.functional as F

#implementation of the contracting path: 2 convolutions + max pooling (2x2, 2) 
class down(nn.Module):
	
	def __init__(self, in_channels, out_channels, dropout=False):
		super(down, self).__init__()
		layers = [
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2(out_channels),
			nn.ReLU(inplace=true),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2(out_channels),
			nn.ReLU(inplace=true),
		]
		if dropout:
			layers.append(nn.Dropout())
		layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
		self.down = nn.Sequential(*layers)

	def forward(self, x):
		x = self.down(x)
		return x

class up(nn.Module):
	
	def __init__(self, in_channels, middle_channels, out_channels):
		super(up, self).__init__()
		self.up = nn.Sequential(
			nn.Conv2d(in_channels, middle_channels, kernel_size=3),
			nn.BatchNorm2d(middle_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
			nn.BatchNorm2d(middle_channels),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
)

	def forward(self, x):
		return self.up(x)

