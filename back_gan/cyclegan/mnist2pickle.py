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
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('dataset', exist_ok = True)

parser = argparse.ArgumentParser(description = "Collecting certain number in mnist")
parser.add_argument('--number', type = int, default =7, help = "mnist number")
parser.add_argument('--batch_size', type = int, default =512, help = "batch_size")

args = parser.parse_args()


train_loader = torch.utils.data.DataLoader(datasets.MNIST('dataset', train= True, download = True, transform = transforms.ToTensor()),
	batch_size = args.batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('dataset', train= False, transform = transforms.ToTensor()), shuffle = True)


result = []
cnt = 0

for i, (image, label) in enumerate(train_loader):
	for index in range(len(label)):

		if label[index].item() == args.number:
			result.append(image[index])
			cnt = cnt + 1
		else:
			continue

	if cnt > 5000: break

with open('dataset/'+str(args.number)+'.pickle', 'wb') as f:
	pickle.dump(result, f)