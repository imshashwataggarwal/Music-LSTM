from __future__ import absolute_import, division, print_function

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets as dsets

import numpy as np
import pandas as pd

from IPython.display import Audio

# librosa - preprocessing : save spectogram as hdf5 files


# Hyperparameters
# represent in an arg dict

# keys = lr, epochs, img_size, img_height, img_width, 
# input_channel, keep_prob, batch_size, z_dim, z_dim_1, z_dim_2

def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)
	

# Model
class Autoencoder(nn.Module):
	
	def __init__(self, arg):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
				nn.Conv2d(arg['input_channel'], 32, kernel_size=3, padding=1),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.Dropout2d(p=arg['keep_prob']),
				
				nn.Conv2d(32, 32, kernel_size=3, padding=1),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.Dropout2d(p=arg['keep_prob']),
            	
            	nn.MaxPool2d(2),

            	nn.Conv2d(32, 64, kernel_size=5, padding=2),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Dropout2d(p=arg['keep_prob']),
				
				nn.Conv2d(64, 64, kernel_size=5, padding=2),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Dropout2d(p=arg['keep_prob']),
            	
            	nn.MaxPool2d(2)
			)

		self.h_dim = (arg['img_height'] // 4) * (arg['img_width'] // 4) * 64
		self.z = nn.Linear(h_dim, arg['z_dim'])
		self.z_dims = (arg['z_dim_1'], arg['z_dim_2'])

		self.decoder = nn.Sequential(
				
				nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Dropout2d(p=arg['keep_prob']),
            	
            	nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=2, output_padding=1),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.Dropout2d(p=arg['keep_prob']),
				
				nn.Conv2d(32, 32, kernel_size=3, padding=1),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.Dropout2d(p=arg['keep_prob']),
            	
            	nn.Conv2d(32, arg['input_channel'], kernel_size=3, padding=1, stride=2, output_padding=1),
				nn.Dropout2d(p=arg['keep_prob']),
				nn.Sigmoid()          	
			)

	def forward(self, x):
		h = self.encoder(x)
		latent = self.z(h)
		h = latent.view(latent.size(0), *self.z_dims, 64)
		out = self.decoder(h)

		return out, latent


# Autoencoder
ae = Autoencoder(arg)
if torch.cuda.is_available():
	ae.cuda()

# Optimizer
optimizer = torch.optim.Adam(ae.parameters(), lr=arg['lr'])


# TODO: Data Loader
# steps = len(data_loader)



# Train the Model
for epoch in range(arg['epochs']):
	for i, spectograms in enumerate(data_loader):
		
		spectograms = to_var(spectograms.view(spectograms.size(0),*arg['img_size']))
		
		# Forward
		out, _ = ae(spectograms)

		# Reconstruction Loss
		loss = F.binary_cross_entropy(out, spectograms, size_average=False)

		# Backward + Optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Logging
		if i%100 == 0:
			print("Epoch [{}/{}], Step [{}/{}], Loss: {}".format(
				epoch+1,arg['epochs'],i+1, steps, loss.data[0]))


# Test the Model
ae.eval()

# spectograms = to_var(spectograms.view(spectograms.size(0),*arg['size']))
fixed_x = # TODO: INPUT SPECTOGRAM

out, latent = ae(fixed_x)
torchvision.utils.save_image(out.data.cpu(), 'Reconstructed_Image.png')


# Save the Trained Model
torch.save(ae.state_dict(), 'ae.pkl')