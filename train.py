import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import BatchSampler
from matplotlib import pyplot as plt
from networks import *
from loss_functions import *
from datasets import *

os.makedirs('checkpoints_FMNIST', exist_ok = True)
os.makedirs('checkpoints_MNIST', exist_ok = True)

train_dataset = MNIST('./', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = MNIST('./', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

#train_dataset = FashionMNIST('./', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
#test_dataset = FashionMNIST('./', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

n_classes = 4
n_samples = 128
batch_size = n_classes * n_samples

mySampler = SequentialSampler(train_dataset)
myBatchSampler = myBatchSampler(mySampler, train_dataset, n_classes = n_classes, n_samples = n_samples)
train_loader = DataLoader(train_dataset, shuffle = False, num_workers = 4, batch_sampler = myBatchSampler)

no_of_training_batches = len(train_loader)/batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 10

embeddingNet = EmbeddingNet()
optimizer = optim.Adam(embeddingNet.parameters(), lr = 3e-4, betas = (0.9, 0.999))

def run_epoch(data_loader, model, optimizer, split = 'train', epoch_count = 0):

	model.to(device)

	if split == 'train':
		model.train()
	else:
		model.eval()

	running_loss = 0.0

	for batch_id, (imgs, labels) in enumerate(train_loader):

		imgs = imgs.to(device)
		embeddings = model.get_embeddings(imgs)
		batch_loss, _ = batch_all_online_triplet_loss(labels, embeddings, margin = 0.2, squared = False)
		optimizer.zero_grad()
		
		if split == 'train':
			batch_loss.backward()
			optimizer.step()

		running_loss = running_loss + batch_loss.item()

	return running_loss

def fit(train_loader, model, optimizer, n_epochs):

	print('Training Started\n')
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(train_loader, model, optimizer, split = 'train', epoch_count = epoch)
		loss = loss/no_of_training_batches

		print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints_MNIST/model_epoch_' + str(epoch + 1) + '.pth')

fit(train_loader, embeddingNet, optimizer = optimizer, n_epochs = epochs)