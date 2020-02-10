import torch
from torch import nn
from torch import optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import pickle
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting data
class MNISTDataset(Dataset):

	def __init__(self, data_path, train, transform):
		self.cls_num, self.content_num = 4, 7



		cls = class2image(self.cls_num)
		cls = torch.stack(cls)
		self.cls = preprocess_color(cls, self.cls_num)

		content = class2image(self.content_num)
		content = torch.stack(content)
		self.content = preprocess_color(content, self.content_num)

		self.result = torch.cat((self.cls, self.content),0)

		self.len = len(self.result)

	def __getitem__(self, index):
		label = self.cls_num if index <= len(self.cls) else self.content_num
		return self.result[index], label

	def __len__(self):
		return self.len
