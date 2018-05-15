import sys
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
import h5py
from PIL import Image
from sklearn.externals import joblib
import shutil
import os
import random
import pickle
import time
import gc
import re
from tensorboardX import SummaryWriter
import time
import math
from torchvision import datasets, models, transforms
import matplotlib.cm as cm
import cv2
import pandas as pd 
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from mosei_dataloader import mosei

preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'---------------------------------------------------LSTM VocalNet-------------------------------------------------------'

class VocalNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers):
		super(VocalNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)

	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		return hiddens[-1]

'---------------------------------------------------Memory to Emotion Decoder------------------------------------------'
class predictor(nn.Module):
	def __init__(self,no_of_emotions,input_size):
		super(predictor, self).__init__()
		self.fc1 = nn.Linear(input_size, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, no_of_emotions)

	def forward(self,x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x
'------------------------------------------------------Hyperparameters-------------------------------------------------'
batch_size = 1
mega_batch_size = 1
no_of_emotions = 6
# vocal_seq_len = 150
# vision_seq_len = 45
use_CUDA = True
use_pretrained =  False
num_workers = 20

test_mode = False
val_mode = False
train_mode = True

no_of_epochs = 1000
vocal_input_size = 74 # Dont Change
vocal_num_layers = 2
vocal_hidden_size = 1024
predictor_input_size = 2048
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = VocalNet(vocal_input_size, vocal_hidden_size, vocal_num_layers)
Predictor = predictor(no_of_emotions,predictor_input_size)
if train_mode:
	train_dataset = mosei(mode= "train")
	data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,num_workers = num_workers)
elif val_mode:
	val_dataset = mosei(mode = "val")
	data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=1,
                                        shuffle=False,num_workers = num_workers)
	no_of_epochs = 1
else:
	test_dataset = mosei(mode = "test")
	data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=1,
                                        shuffle=False,num_workers = num_workers)
	no_of_epochs = 1
curr_epoch = 0
total = 0
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = Vocal_encoder.cuda()
Predictor = Predictor.cuda()
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.MSELoss(size_average = False)
params =  list(Vocal_encoder.parameters())+list(Predictor.parameters())
print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, lr = 0.0001)
# optimizer = torch.optim.SGD(params, lr =0.001,momentum = 0.9 )

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='vocal_net'):
	filename = filename +'_'+str(state['epoch'])+'.pth.tar'
	os.system("mkdir -p vocal_only") 
	torch.save(state, './vocal_only/'+filename)
	if is_final:
		shutil.copyfile(filename, 'model_final.pth.tar')


'-------------------------------------------Setting into train mode----------------------------------------------------'

if not train_mode:
	Vocal_encoder.train(False)
	Predictor.train(False)
else:
	Vocal_encoder.train(True)
	Predictor.train(True)

'----------------------------------------------------------------------------------------------------------------------'
epoch = 0
y_true = []
y_pred = []
while epoch<no_of_epochs:
	j_start = 0
	running_loss = 0
	running_corrects = 0
	if use_pretrained:
		# pretrained_file = './vocal_only/vocal_net_iter_8000_0.pth.tar'
		pretrained_file = './vocal_only/vocal_net__0.pth.tar'

		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		Predictor.load_state_dict(checkpoint['Predictor'])
		use_pretrained = False
		if train_mode:
			epoch = checkpoint['epoch']+1
			optimizer.load_state_dict(checkpoint['optimizer'])

	K = 0
	for i,(vision,vocal,emb,gt) in enumerate(data_loader):
		if use_CUDA:
			vocal = Variable(vocal.float()).cuda()
			gt = Variable(gt.float()).cuda()

		vocal_output = Vocal_encoder(vocal)
		outputs = Predictor(output)
		outputs = torch.clamp(outputs,0,3)
		loss = criterion(outputs, gt)
		if train_mode and K%mega_batch_size==0:
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			Vocal_encoder.zero_grad()
			Predictor.zero_grad()

		# outputs_ = Variable(torch.FloatTensor([ 0.1565 ,0.1233,  0.0401,  0.4836 , 0.1596,  0.04842])).cuda()
		# loss = criterion(outputs_, gt)

		running_loss += loss.data[0]
		K+=1
		average_loss = running_loss/K
		if train_mode and K%mega_batch_size==0:			
			print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f'
			% (epoch+1, K, average_loss))
		elif val_mode:
			print('Validating -- Epoch [%d], Sample [%d], Average Loss: %.4f'
			% (epoch+1, K, average_loss))
		elif test_mode:
			print('Testing -- Epoch [%d], Sample [%d], Average Loss: %.4f'
			 % (epoch+1, K, average_loss))

		if train_mode:
			if K%4000==0:
				save_checkpoint({
					'epoch': epoch,
					'loss' : running_loss,
					'j_start' : 0,
					'Vocal_encoder': Vocal_encoder.state_dict(),
					'Predictor' : Predictor.state_dict(),
					'optimizer': optimizer.state_dict(),
				}, False,'vocal_net_iter_'+str(K))
	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'loss' : running_loss,
			'j_start' : 0,
			'Vocal_encoder': Vocal_encoder.state_dict(),
			'Predictor' : Predictor.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, False,'vocal_net_')
	epoch+= 1 
'------------------------------------------------------Saving model after training completion--------------------------'
if train_mode:
	save_checkpoint({
		'epoch': epoch,
		'loss' : running_loss,
		'j_start' : 0,
		'Vocal_encoder': Vocal_encoder.state_dict(),
		'Predictor' : Predictor.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, False)

# print('Accuracy:', accuracy_score(y_true, y_pred))
# print('F1 score:', f1_score(y_true, y_pred,average = 'weighted'))
# print('Recall:', recall_score(y_true, y_pred,average ='weighted'))
# print('Precision:', precision_score(y_true, y_pred,average = 'weighted'))
