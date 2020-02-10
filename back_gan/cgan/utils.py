import cv2
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
import os
import matplotlib
import pickle

color_classes = {0: [255, 0, 0], 1: [232, 235, 52], 2: [235, 150, 52], 3: [180, 255, 0], 4: [0, 0, 255], 5: [0, 180, 255], 6: [0,0,255], 7: [255, 0, 0], 8: [252, 0, 194], 9: [180, 180, 180]}
#color_classes = {0: [255, 0, 0], 1: [232, 235, 52], 2: [235, 150, 52], 3: [180, 255, 0], 4: [150, 75, 0], 5: [0, 180, 255], 6: [0,0,255], 7: [0, 0, 0], 8: [252, 0, 194], 9: [180, 180, 180]}
#color_classes = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

## test_image부터 시도
os.makedirs('sample_image', exist_ok = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting preprocess
def preprocess_4b(real_images, labels):
	black = torch.zeros(real_images.shape[1:]).to(device)
	black_row = torch.cat((black, black), 2).to(device)
	result = [] 
	for i in range(labels.shape[0]):
		label = labels[i].item()
		if label == 0 or label == 4 or label == 8:
			image_row = torch.cat((real_images[i], black), 2)
			temp = torch.cat((image_row, black_row), 1)
			
		elif label == 1 or label == 5 or label == 9:
			image_row = torch.cat((black, real_images[i]), 2)
			temp = torch.cat((image_row, black_row), 1)
	
		elif label == 2 or label== 6:
			image_row = torch.cat((real_images[i], black), 2)
			temp = torch.cat((black_row, image_row), 1)

		elif label == 3 or label ==7:
			image_row = torch.cat((black, real_images[i]), 2)
			temp = torch.cat((black_row, image_row), 1)
		
		else:
			print("unknown label")
			exit(0)
		result.append(temp)
	result = torch.stack(result)
	result = result.to(device)
	return result

def preprocess_1b(real_images, labels):
	black = torch.zeros(real_images.shape[1:]).to(device)
	black_row = torch.cat((black, black), 2).to(device)
	result = [] 
	for i in range(labels.shape[0]):
		label = labels[i].item()
		image_row = torch.cat((real_images[i], black), 2)
		temp = torch.cat((image_row, black_row), 1)	
		result.append(temp)
	result = torch.stack(result)
	result = result.to(device)
	return result

def preprocess_color(real_image, labels):
	colored_image = real_image.cpu().numpy()
	colored_image = np.repeat(colored_image, 3, axis = 1)
	plot_list = []
	noise_list = []	
	
	for index in range(real_image.shape[0]):
		for i in range(3):
			channel = colored_image[index, i, :, :]
			mask_x, mask_y = np.where(channel == 0)
			if type(labels) == int:
				color = color_classes[labels][i]
			else:
				color = color_classes[labels[index].item()][i]
			colored_image[index,i][mask_x, mask_y] =  color / 255
			result = colored_image[index]
		plot_list.append(result)
	save_image(torch.tensor(plot_list), 'sample_image/result1.png', nrow= 8)
	colored_image = torch.from_numpy(colored_image).float().to(device)
	return colored_image


def add_color(real_image, labels):
	colored_image = real_image.cpu().numpy()
	labels = labels.cpu().numpy()
	colored_image = np.repeat(colored_image, 3, axis = 1)
	plot_list = []
	noise_list = []	
	
	for index in range(real_image.shape[0]):
		for i in range(3):
			channel = colored_image[index, i, :, :]
			mask_x, mask_y = np.where(channel == 0)
			colored_image[index,i][mask_x, mask_y] =  color_classes[(int(labels[index])+ 2) % 10][i] / 255
			result = colored_image[index]
		plot_list.append(result)
	save_image(torch.tensor(plot_list), 'sample_image/result2.png', nrow= 8)
	colored_image = torch.from_numpy(colored_image).float().to(device)
	return colored_image

def add_random_color(real_image, labels):
	colored_image = real_image.cpu().numpy()
	labels = labels.cpu().numpy()
	colored_image = np.repeat(colored_image, 3, axis = 1)
	plot_list = []
	noise_list = []	
	
	for index in range(real_image.shape[0]):
		for i in range(3):
			channel = colored_image[index, i, :, :]
			mask_x, mask_y = np.where(channel == 0)
			colored_image[index,i][mask_x, mask_y] =  color_classes[np.random.randint(10)][i] / 255
			result = colored_image[index]
		plot_list.append(result)
	save_image(torch.tensor(plot_list), 'sample_image/result3.png', nrow= 8)
	colored_image = torch.from_numpy(colored_image).float().to(device)
	return colored_image


def label2vec(labels):
	result = torch.zeros(labels.shape[0], 2).float()
	for i in range(result.shape[0]):
		if labels[i].item() == 4:
			result[i][0] = 1.0
		elif labels[i].item() == 7:
			result[i][1] = 1.0
		else:
			print('unknown number')
			exit(0)
	
	return result

def class2image(number):
	with open('dataset/'+str(number)+'.pickle', 'rb') as file:
		data = pickle.load(file)
	return data


def color2vec(labels):
	result = torch.zeros(labels.shape[0], 2).float()
	
	for i in range(result.shape[0]):
		result[i][(labels[i].item()+ 1) % 2] = 1.0
	
	return result

def valid(data):
	return torch.ones(data.size())

def fake(data):
	return torch.zeros(data.size())





