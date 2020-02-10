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


class Generator(nn.Module):
	def __init__(self, out_layer):
		super(Generator,self).__init__()
		self.layer1 = nn.Linear(100 + 2 + 2, out_layer)
		self.layer2 = nn.Linear(out_layer, out_layer * 2)
		self.layer3 = nn.Linear(out_layer * 2, out_layer * 4)
		self.layer4 = nn.Linear(out_layer * 4, out_layer * 8)
		self.layer5 = nn.Linear(out_layer * 8, out_layer * 16)
		self.layer6 = nn.Linear(out_layer * 16, 784 * 3)
		self.sigmoid = nn.Sigmoid()
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.tanh = nn.Tanh()

	def forward(self, data, condition, color):
		data = data.view(data.shape[0], -1)
		condition = condition.float()
		data = torch.cat((data,  condition, color), 1)
		
		x = self.leakyrelu(self.layer1(data))
		x = self.leakyrelu(self.layer2(x))
		x = self.leakyrelu(self.layer3(x))
		x = self.leakyrelu(self.layer4(x))
		x = self.leakyrelu(self.layer5(x))
		x = self.leakyrelu(self.layer6(x))
		x = self.tanh(x)
		return x

class Discriminator(nn.Module):
	def __init__(self, out_layer):
		super(Discriminator,self).__init__()
		self.layer1 = nn.Linear(3 * 784 + 2 + 2, out_layer * 8)
		self.layer2 = nn.Linear(out_layer * 8, out_layer * 4)
		self.layer3 = nn.Linear(out_layer * 4, out_layer * 2)
		self.layer4 = nn.Linear(out_layer * 2, out_layer)
		self.layer5_cls = nn.Linear(out_layer, 10)
		self.layer5_valid = nn.Linear(out_layer, 1)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax()
		self.leakyrelu = nn.LeakyReLU(0.2)

	def forward(self, data, condition, color):
		data = data.view(data.shape[0], -1)
		condition = condition.float()
		data = torch.cat((data, condition, color), 1)
		x = self.leakyrelu(self.layer1(data))
		x = self.leakyrelu(self.layer2(x))
		x = self.leakyrelu(self.layer3(x))
		x = self.leakyrelu(self.layer4(x))
		cls = self.softmax(self.layer5_cls(x))
		validity = self.leakyrelu(self.layer5_valid(x))
		validity = self.sigmoid(validity)
		return validity, cls

class ConvClassifier(nn.Module):
	def __init__(self):
		super(ConvClassifier, self).__init__()
		self.conv1 = nn.Conv2d(3,16,5) #Three-channel
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(16, 32 ,5)
		self.fc1 = nn.Linear(32*4*4,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, x):
		x= self.pool(F.relu(self.conv1(x)))
		x= self.pool(F.relu(self.conv2(x)))
		x= x.view(-1, 32*4*4)
		x= F.relu(self.fc1(x))
		x= F.relu(self.fc2(x))
		x= self.fc3(x)
		return x

