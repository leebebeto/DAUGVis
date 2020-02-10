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
#color_classes = {0: [255, 0, 0], 1: [232, 235, 52], 2: [235, 150, 52], 3: [180, 255, 0], 4: [150, 75, 0], 5: [0, 180, 255], 6: [0,0,255], 7: [0, 0, 0], 8: [252, 0, 194], 9: [180, 180, 180]}
color_classes = {0: [255, 0, 0], 1: [232, 235, 52], 2: [235, 150, 52], 3: [180, 255, 0], 4: [0, 0, 255], 5: [0, 180, 255], 6: [0,0,255], 7: [255, 0, 0], 8: [252, 0, 194], 9: [180, 180, 180]}
os.makedirs('sample_image', exist_ok = True)

def valid(data):
	return torch.ones(data.size())

def fake(data):
	return torch.zeros(data.size())

def class2image(number):
	with open('dataset/'+str(number)+'.pickle', 'rb') as file:
		data = pickle.load(file)
	return data


def preprocess_color(real_image, labels):
	colored_image = real_image.cpu().numpy()
	colored_image = np.repeat(colored_image, 3, axis = 1)
	plot_list = []
	noise_list = []	
	
	for index in range(real_image.shape[0]):
		for i in range(3):
			channel = colored_image[index, i, :, :]
			mask_x, mask_y = np.where(channel == 0)
			colored_image[index,i][mask_x, mask_y] =  color_classes[labels][i] / 255
			result = colored_image[index]
		plot_list.append(result)
	save_image(torch.tensor(plot_list), 'sample_image/result1.png', nrow= 8)
	colored_image = torch.from_numpy(colored_image).float().to(device)
	return colored_image

