from u_net_parts import *
import torch.nn.functional as F

class UNet(nn.Module):

	def __init__(self, num_channels, num_classes):
		super(UNet, self).__init__()
		self.down1 = down(3, 64)
		self.down2 = down(64,128)
		self.down3 = down(128, 256)
		self.down4 = down(256, 512, dropout=True)
		self.center = up(512, 1024, 512)
		self.up4 = up(1024, 512, 256)
		self.up3 = up(512, 256, 128)
		self.up2 = up(256, 128, 64)
		self.up1 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3),
			nn.ReLU(inplace=True),
		)
		self.final = nn.Conv2d(64, num_classes, kernel_size=1)
		initialize_weights(self)
			
	def forward(self, x):
		x1 = self.down(x)
		x2 = self.down1(x1)
		x3 = self.down3(x2)
		x4 = self.down4(x3) 
		center = self.center(x4)
		
		y4 = self.up4(torch.cat([center, F.upsample(x4, center.size()[2:], mode='bilinear')], 1))
		y3 = self.up3(torch.cat([y4, F.upsample(x3, y4.size()[2:], mode='bilinear')], 1))
		y2 = self.up2(torch.cat([y3, F.upsample(x2, y3.size()[2:], mode='bilinear')], 1))
		y1 = self.up1(torch.cat([y2, F.upsample(x1, y2.size()[2:], mode='bilinear')], 1))
		final = self.final(y1)
		return F.upsample(final, x1.size()[2:], mode='bilinear')
