import torch
from torch import nn
from torch import optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchsummary import summary
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
from model import *
from data import *
from utils import *
import itertools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description = "CycleGAN")
parser.add_argument('--batch_size', type = int, default =64, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 10, help = "epoch")
parser.add_argument('--learning_rate', type = float, default = 0.0002, help = "learning_rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_recon", type=float, default = 2.5, help="reconstruction lambda")
parser.add_argument("--lambda_identity", type=float, default=5.0, help="identity lambda")
parser.add_argument("--out_nc", type=int, default=64, help="number of output channels")
args = parser.parse_args()

print("device", device)
os.makedirs('generated-images', exist_ok = True)

# setting data
train_data = MNISTDataset('dataset', train = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)

test_data = MNISTDataset('dataset', train = False, transform = transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, shuffle = True)


G_A2B = Generator(args.out_nc).to(device)
G_B2A = Generator(args.out_nc).to(device)
D_A = Discriminator(args.out_nc).to(device)
D_B = Discriminator(args.out_nc).to(device)
g_optimizer = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr = args.learning_rate, betas = (args.b1, args.b2))
d_A_optimizer = optim.Adam(D_A.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))
d_B_optimizer = optim.Adam(D_B.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))

criterion_gan = nn.MSELoss()
criterion_identity = nn.L1Loss()
criterion_recon = nn.L1Loss()

# train
G_A2B.train()
G_B2A.train()
D_A.train()
D_B.train()

for iteration in range(args.epoch):
	for i, (A, B) in enumerate(train_loader):

		real_A = Variable(A.to(device))
		real_B = Variable(B.to(device))


		g_optimizer.zero_grad()
		# train generator
		loss_identity_A = criterion_identity(G_B2A(real_A), real_A)
		loss_identity_B = criterion_identity(G_A2B(real_B), real_B)

		loss_identity = (loss_identity_A + loss_identity_B) / 2

		fake_A = G_B2A(real_B)
		fake_B = G_A2B(real_A)

		loss_gen_A = criterion_gan(fake_A, valid(fake_A))
		loss_gen_B = criterion_gan(fake_B, valid(fake_A))

		loss_gen = (loss_gen_A + loss_gen_B) / 2

		recon_A = G_B2A(fake_B)
		recon_B = G_A2B(fake_A)

		loss_recon_A = criterion_recon(recon_A, real_B)
		loss_recon_B = criterion_recon(recon_B, real_B)
		loss_recon = (loss_recon_A + loss_recon_B) / 2

		loss_g = loss_gen + args.lambda_identity * loss_identity + args.lambda_recon * loss_recon

		loss_g.backward()
		g_optimizer.step()

		# train discriminator
		
		d_A_optimizer.zero_grad()
		loss_A_real = criterion_gan(real_A, valid(real_A)) 
		loss_A_fake = criterion_gan(fake_A.detach(), fake(fake_A))
		loss_discriminator_A = (loss_A_real + loss_A_fake) / 2

		loss_discriminator_A = Variable(loss_discriminator_A, requires_grad = True)
		loss_discriminator_A.backward()
		d_A_optimizer.step()

		d_B_optimizer.zero_grad()
		loss_B_real = criterion_gan(real_B, valid(real_B)) 
		loss_B_fake = criterion_gan(fake_B.detach(), fake(fake_B))
		loss_discriminator_B = (loss_B_real + loss_B_fake) / 2

		loss_discriminator_B = Variable(loss_discriminator_B, requires_grad = True)
		loss_discriminator_B.backward()
		d_B_optimizer.step()

		loss_d = (loss_discriminator_A + loss_discriminator_B) / 2

		loss = loss_g + loss_d 


		if i % 50 == 0:	
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f} , G-R-Loss: {:.4f} , G-I-Loss: {:.4f} , G-Adv-Loss: {:.4f}, D-Loss: {:.4f}, D-A-Loss: {:.4f}, D-B-Loss: {:.4f} '.format(iteration+1, args.epoch, i, int(len(train_loader)), loss.item(), loss_g, loss_recon, loss_identity, loss_gen, loss_d, loss_discriminator_A, loss_discriminator_B))
		if i % 20 == 0:	
			plot_list = []
			for image in [real_A[0], real_B[0], fake_A[0], fake_B[0]]:
				image = image.detach().numpy()
				plot_list.append(image)
			save_image(torch.tensor(plot_list), 'generated-images/'+str(iteration+1)+'-'+str(i)+'.png', nrow=4)


