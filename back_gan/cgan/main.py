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
from data import *
# test_image부터 시도

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('generated-images', exist_ok = True)

# setting args
parser = argparse.ArgumentParser(description = "Conditional Generative Adversarial Network using MNIST dataset")
parser.add_argument('--batch_size', type = int, default = 64, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 250, help = "epoch")
parser.add_argument('--g_learning_rate', type = float, default = 0.0002, help = "generator learning_rate")
parser.add_argument('--d_learning_rate', type = float, default = 0.0002, help = "discriminator learning_rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_vector", type=int, default=100, help="latent dimension")
parser.add_argument("--out_dim", type=int, default=256, help="number of output linear dim")

args = parser.parse_args()


print("device", device)
print(args)
# setting data

train_data = MNISTDataset('dataset', train = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)

test_data = MNISTDataset('dataset', train = False, transform = transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, shuffle = True)


generator = Generator(args.out_dim).to(device)
discriminator = Discriminator(args.out_dim).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr = args.g_learning_rate, betas = (args.b1, args.b2))
d_optimizer = optim.Adam(discriminator.parameters(), lr = args.d_learning_rate, betas = (args.b1, args.b2))

criterion_gan = nn.BCELoss()
criterion_recon = nn.L1Loss()
criterion_cls = nn.CrossEntropyLoss()


# train
generator.train()
discriminator.train()


for epoch in range(args.epoch):
	for i, (images, labels) in enumerate(train_loader):
		labels = labels.to(device)
		real_images = images.to(device)
		condition = label2vec(labels).to(device)
		color = color2vec(labels).to(device)
		
		# real_images = preprocess_color(images, labels)
		random_vector = torch.randn(real_images.shape[0],args.latent_vector).to(device)
		if real_images.shape[0] < args.batch_size: break 
		
		generated_image = generator(random_vector, condition, color).view(real_images.shape[0], 3, 28, 28)

		real_valid, real_cls = discriminator(real_images, condition, color)
		fake_valid, fake_cls = discriminator(generated_image, condition, color)

		# train discriminator		
		d_loss_adv_real = criterion_gan(real_valid, valid(real_valid).to(device))
		d_loss_adv_fake = criterion_gan(fake_valid, fake(fake_valid).to(device))
		d_loss_adv = d_loss_adv_real + d_loss_adv_fake

		d_loss_cls = criterion_cls(real_cls, labels)

		d_loss = d_loss_adv + d_loss_cls

		d_optimizer.zero_grad()
		d_loss.backward(retain_graph = True)
		d_optimizer.step()
		

		# train generator
		g_loss_adv = criterion_gan(fake_valid, valid(fake_valid).to(device))
		g_loss_cls = criterion_cls(fake_cls, labels)
		g_recon_loss = criterion_recon(generated_image, real_images)
		g_loss = g_loss_adv + g_loss_cls + 0.25 * g_recon_loss
		
#		g_loss = g_gan_loss + 5 * g_recon_loss
		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()
		
		loss = d_loss / (d_loss + g_loss)


		if i % 50 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f}, D-Loss: {:.4f} '.format(epoch+1, args.epoch, i, int(len(train_loader)), loss.item(), g_loss, d_loss))
		if i % 50 == 0:
			plot_list = []
			for iter in range(16):
				if iter > 32: break
				test = generated_image[iter]
				test = test.view(3,28,28)
				npimg = test.detach().numpy()
#				npimg = test.detach().cpu().numpy()
				plot_list.append(npimg)
			save_image(torch.tensor(plot_list), 'generated-images/'+str(epoch+1)+'-'+str(i)+'.png', nrow=4)
	if epoch % 10 == 0:
		torch.save(generator.state_dict(), 'check_point/generator_'+str(epoch)+'.pt')
		torch.save(discriminator.state_dict(), 'check_point/discriminator.pt')

		

