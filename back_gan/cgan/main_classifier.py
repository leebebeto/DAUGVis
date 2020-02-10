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
from test import *
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
parser.add_argument('--threshold', type = int, default = 56, help = 'index where to add random color')
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


# train
model_conv.train()

for epoch in range(args.num_epochs):
	for i, (images, labels) in enumerate(train_loader):

		labels = labels.to(device)
		if labels.shape[0] < args.batch_size: break
		
		#images_origin = images[:args.threshold]
		#labels_origin = labels[:args.threshold]
		#images_add = images[args.threshold:]
		#labels_add = labels[args.threshold:]
		
		images_origin = preprocess_color(images_origin, labels_origin)
		#images_add = add_random_color(images_add, labels_add)
		#if i == 900:
		#	images_add = get_generated(args.batch_size - args.threshold, args.check_epoch, labels_add, True)
		#else:
		#	images_add = get_generated(args.batch_size - args.threshold, args.check_epoch, labels_add, False)
		#real_images = torch.cat((images_origin, images_add), 0)
		#images = real_images.to(device)
		images = images_origin.to(device)
		outputs = model_conv(images)
		loss = criterion(outputs, labels)

		#Backward and optimize
		optimizer_conv.zero_grad()
		loss.backward()
		optimizer_conv.step()

		if(i+1) % 64 ==0:
			correct, total = 0, 0
			print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'
				. format(epoch+1, args.num_epochs, i+1, len(train_loader), loss.item()))
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0) # + batch_size
			correct += (predicted == labels).sum()

			print('Accuracy on the test images: {:.4f} %'.format(100*correct / total))

print('Finished Training!')
if args.save_model:
	torch.save(model_conv.state_dict(), 'check_point/MNIST_Classifier_none_'+str(args.check_epoch)+'.pt')



