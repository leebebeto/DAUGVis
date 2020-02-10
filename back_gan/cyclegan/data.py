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

parser = argparse.ArgumentParser(description = "Deep Convolutional Generative Adversarial Network using CIFAR-10 dataset")
parser.add_argument('--batch_size', type = int, default =128, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 10, help = "epoch")
parser.add_argument('--learning_rate', type = float, default = 0.0002, help = "learning_rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_vector", type=int, default=100, help="latent dimension")
args = parser.parse_args()


# setting data
class MNISTDataset(Dataset):

	def __init__(self, data_path, train, transform):
		cls_num, content_num = 4, 7

		cls = class2image(cls_num)
		cls = torch.stack(cls)
		cls = preprocess_color(cls, cls_num)

		content = class2image(content_num)
		content = torch.stack(content)
		content = preprocess_color(content, content_num)

		self.cls = cls
		self.content = content

		self.len = len(cls)

	def __getitem__(self, index):
		return self.cls[index % 5000], self.content[index % 5000]

	def __len__(self):
		return self.len
