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
# test_image부터 시도

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('generated-images', exist_ok = True)

# setting args
parser = argparse.ArgumentParser(description = "Classification using MNIST dataset")
parser.add_argument('--batch_size', type = int, default = 64, help = "input batch size for training (default = 64)")
parser.add_argument('--num_epochs', type=int, default = 1, help ="number of epochs to train(default = 5)")
parser.add_argument('--lr', type = float, default = 0.005, help="learning rate (default = 0.005)")
parser.add_argument('--no_cuda', action ='store_true', default = False, help = "disable CUDA training")
parser.add_argument('--seed', type = int, default =1, help='random seed (default = 1)')
parser.add_argument('--save_model', action = 'store_true', default = True, help = 'for saving current model')
parser.add_argument('--check_epoch', type = int, default = 0, help = 'epoch of check point')
args = parser.parse_args()


print("device", device)
print(args)
# setting data
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train= True, download = True, transform = transforms.ToTensor()),
	batch_size = args.batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train= False, transform = transforms.ToTensor()), shuffle = True)


model_conv = ConvClassifier().to(device)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.Adam(model_conv.parameters(), lr = args.lr)
model_conv.load_state_dict(torch.load('check_point/MNIST_Classifier_none_'+str(args.check_epoch)+'.pt'))

# train
with torch.no_grad():
	correct = 0
	total = 0
	for i, (images, labels) in enumerate(test_loader):
		labels = labels.to(device)
		
		#labels_temp = labels.detach().cpu()
		#mask = np.isin(labels_temp, [0,1,2,3])
		
		#labels = labels[mask]
		#images = images[mask]
		
		
		real_images = preprocess_color(images, labels)
		

		outputs = model_conv(real_images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0) # + batch_size
		correct += (predicted == labels).sum()
	print('Without unseen data: Accuracy on the test images: {:.2f} %'.format(100*correct / total))
	
	for i, (images, labels) in enumerate(test_loader):
		labels = labels.to(device)
		real_images = add_random_color(images, labels)	

		outputs = model_conv(real_images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0) # + batch_size
		correct += (predicted == labels).sum()
	print('With unseen data: Accuracy on the test images: {:.2f} %'.format(100*correct / total))
	



