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
from utils import *
from model import *

#parser = argparse.ArgumentParser(description = "test generator")
#parser.add_argument('--epoch', type = int, default = 50, help = 'epoch of checkpoint to test')
#parser.add_argument('--len', type = int, default = 16, help = 'number of generated images')
#args = parser.parse_args() 
epoch = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_generated(length, epoch, labels, save):

generator = Generator(256).to(device)
check_point = torch.load('check_point/generator_'+str(epoch)+'.pt')
generator.load_state_dict(check_point)

labels = torch.tensor([4,7,4,7,4,4,4,4,7,7,7,7,4,7,4,7])
length = len(labels)
print(length)
with torch.no_grad():
	test_list = []
	return_list = []
	for i in range(length):
		curr_label = labels[i]
		condition= torch.zeros(1,2).to(device)
		color = torch.zeros(1,2).to(device)
		if curr_label == 4:
			condition[0][0] = 1
			color[0][0] = 1
		else:
			condition[0][1] = 1
			color[0][1] = 1
		test_image = torch.randn(1, 100).to(device)
		test_image = generator(test_image, condition, color).view(3,28,28)
		return_list.append(test_image)
		test_image = test_image.detach().numpy()
		test_list.append(test_image)
	# if save == True:
	# 	save_image(torch.tensor(test_list), 'final-test-image_'+str(epoch)+'_'+str(labels[0].item())+'_'+str(labels[1].item())+'_'+str(labels[2].item())+'_'+str(labels[3].item())+'.png', nrow=4)
	# else:
	save_image(torch.tensor(test_list), 'final-test-image_'+str(epoch)+'.png', nrow=4)

	# return torch.stack(return_list)

#labels = torch.tensor([0,1,2,3]).to(device)

#_ = get_generated(args.len, args.epoch, labels)

