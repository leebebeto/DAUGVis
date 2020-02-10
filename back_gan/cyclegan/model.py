import torch
from torch import nn
from torch import optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchsummary import summary
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
	def __init__(self, out_ch):
		super(ResidualBlock,self).__init__()

		self.layer = nn.Sequential(
								   nn.Conv2d(out_ch, out_ch, 1),
								   nn.InstanceNorm2d(out_ch),
								   nn.ReLU(),
								   nn.Conv2d(out_ch, out_ch, 1),
								   nn.InstanceNorm2d(out_ch))
	def forward(self, x):
		return x + self.layer(x)

class Generator(nn.Module):
	def __init__(self, out_ch):
		super(Generator, self).__init__()
		self.down_layer1 = nn.Sequential(nn.Conv2d(3,out_ch,3,2),
									nn.InstanceNorm2d(out_ch),
									nn.ReLU(inplace = True))
		self.down_layer2 = nn.Sequential(nn.Conv2d(out_ch,out_ch*2,3,2),
									nn.InstanceNorm2d(out_ch*2),
									nn.ReLU(inplace = True))
		self.down_layer3 = nn.Sequential(nn.Conv2d(out_ch*2, out_ch*4,3,2),
									nn.InstanceNorm2d(out_ch*4),
									nn.ReLU(inplace = True))
		
		self.up_layer1 = nn.Sequential(nn.ConvTranspose2d(out_ch*4, out_ch*2,4,2),
									nn.InstanceNorm2d(out_ch*2),
									nn.ReLU(inplace = True))

		self.up_layer2 = nn.Sequential(nn.ConvTranspose2d(out_ch*2, out_ch,4,2),
									nn.InstanceNorm2d(out_ch),
									nn.ReLU(inplace = True))

		self.up_layer3 = nn.Sequential(nn.ConvTranspose2d(out_ch, 3,2,2),
									nn.InstanceNorm2d(3),
									nn.ReLU(inplace = True))

		self.residual = ResidualBlock(out_ch*4)

	def forward(self, data):

		x = self.down_layer1(data)
		x = self.down_layer2(x)
		x = self.down_layer3(x)

		for _ in range(6):
			x = self.residual(x)

		x = self.up_layer1(x)
		x = self.up_layer2(x)
		x = self.up_layer3(x)

		return x


class Discriminator(nn.Module):
	def __init__(self, out_ch):
		super(Discriminator,self).__init__()
		self.layer1 = nn.Sequential(nn.Conv2d(3, out_ch, 4, 2),
									nn.LeakyReLU(0.2))
		self.layer2 = nn.Sequential(nn.Conv2d(out_ch, out_ch*2, 4, 2),
									nn.InstanceNorm2d(out_ch*2),
									nn.LeakyReLU(0.2))
		self.layer3 = nn.Sequential(nn.Conv2d(out_ch*2, out_ch*4, 4, 2),
									nn.InstanceNorm2d(out_ch*4),
									nn.LeakyReLU(0.2))
		self.layer4 = nn.Sequential(nn.Conv2d(out_ch*4, out_ch*8, 4, 2),
									nn.InstanceNorm2d(out_ch*8),
									nn.LeakyReLU(0.2))
		self.layer5 = nn.Conv2d(out_ch*8, 1, 1)

	def forward(self, data):
		import pdb; pdb.set_trace()
		x = self.layer1(data)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = x.view(-1,1)
		return x

