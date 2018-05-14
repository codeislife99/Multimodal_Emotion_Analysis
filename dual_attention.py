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
		self.linear = nn.Linear(hidden_size, 6)


	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		hiddens = hiddens.squeeze(1)
		return hiddens
'---------------------------------------------------LSTM VisualNet-------------------------------------------------------'

class VisionNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers):
		super(VisionNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)


	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		hiddens = hiddens.squeeze(1)
		return hiddens


'---------------------------------------------------Gated Attention----------------------------------------------------'

class GatedAttention(nn.Module):
	def __init__(self,att_input_size,att_hidden_size,att_num_layers,no_of_emotions):
		super(GatedAttention, self).__init__()
		self.lstm = nn.LSTM(att_input_size,att_hidden_size,att_num_layers)
		self.linear = nn.Linear(att_hidden_size, no_of_emotions)


	def forward(self,vocal,vision):

		vocal = vocal.repeat(45,1)
		vision = vision.squeeze(2)
		vision = vision.squeeze(2)
		fusion = vocal*vision
		fusion = fusion.unsqueeze(0)
		fusion = fusion.transpose(0,1)
		hiddens,_ = self.lstm(fusion)
		outputs = self.linear(hiddens[-1])
		return outputs	
'---------------------------------------------------Dual Attention----------------------------------------------------'

class DualAttention(nn.Module):
	def __init__(self,no_of_emotions,dan_hidden_size):
		super(DualAttention, self).__init__()
		N = dan_hidden_size
		''' K= 1 ''' 
		self.Wvision_1 = nn.Linear(N,N)
		self.Wvision_m1 = nn.Linear(N,N)
		self.Wvision_h1 = nn.Linear(N,N)
		self.Wvocal_1 = nn.Linear(N,N)
		self.Wvocal_m1 = nn.Linear(N,N)
		self.Wvocal_h1 = nn.Linear(N,N)

		''' K = 2 '''
		self.Wvision_2 = nn.Linear(N,N)
		self.Wvision_m2 = nn.Linear(N,N)
		self.Wvision_h2 = nn.Linear(N,N)
		self.Wvocal_2 = nn.Linear(N,N)
		self.Wvocal_m2 = nn.Linear(N,N)
		self.Wvocal_h2 = nn.Linear(N,N)

		self.fc = nn.Linear(N, no_of_emotions)


	def forward(self,vocal,vision):
		# Sorting out vision
		# print(resnet_output.size())
		# resnet_output = resnet_output.mean(0)
		# resnet_output = resnet_output.view(512,49)
		# vision = resnet_output.transpose(0,1)

		'-------------------------------------------------Initializing Memory--------------------------------------'

		vision_zero = vision.mean(0).unsqueeze(0)
		vocal_zero = vocal.mean(0).unsqueeze(0)
		m_zero = vision_zero * vocal_zero 
		m_zero_vision = m_zero.repeat(vision.size(0),1)
		m_zero_vocal = m_zero.repeat(vocal.size(0),1)
		'--------------------------------------------------K = 1 ---------------------------------------------------'
		# Visual Attention
		h_one_vision = F.tanh(self.Wvision_1(vision))*F.tanh(self.Wvision_m1(m_zero_vision))
		a_one_vision = F.softmax(self.Wvision_h1(h_one_vision),dim=-1)
		vision_one = (a_one_vision*vision).mean(0).unsqueeze(0)

		# Vocal Attention
		h_one_vocal = F.tanh(self.Wvocal_1(vocal))*F.tanh(self.Wvocal_m1(m_zero_vocal))
		a_one_vocal = F.softmax(self.Wvocal_h1(h_one_vocal),dim=-1)
		vocal_one = (a_one_vocal*vocal).mean(0).unsqueeze(0)

		# Memory Update
		m_one = m_zero + vision_one * vocal_one 
		m_one_vision = m_one.repeat(vision.size(0),1)
		m_one_vocal = m_one.repeat(vocal.size(0),1)

		'--------------------------------------------------K = 2  ---------------------------------------------------'

		# Visual Attention
		h_two_vision = F.tanh(self.Wvision_2(vision))*F.tanh(self.Wvision_m2(m_one_vision))
		a_two_vision = F.softmax(self.Wvision_h2(h_two_vision),dim=-1)

		vision_two = (a_two_vision*vision).mean(0).unsqueeze(0)
		# Vocal Attention
		h_two_vocal = F.tanh(self.Wvocal_2(vocal))*F.tanh(self.Wvocal_m2(m_one_vocal))
		a_two_vocal = F.softmax(self.Wvocal_h2(h_two_vocal),dim=-1)
		vocal_two = (a_two_vocal*vocal).mean(0).unsqueeze(0)

		# Memory Update
		m_two = m_one + vision_two * vocal_two 
		return m_two
		'-------------------------------------------------Prediction--------------------------------------------------'
		# return m_two
		# outputs = self.fc(m_two)
		# # print(outputs)
		# return outputs	
'---------------------------------------------------Memory to Emotion Decoder------------------------------------------'
class predictor(nn.Module):
	def __init__(self,no_of_emotions,hidden_size):
		super(predictor, self).__init__()
		self.fc = nn.Linear(hidden_size, no_of_emotions)
	def forward(self,x):
		x = self.fc(x)
		return x
'------------------------------------------------------Hyperparameters-------------------------------------------------'
batch_size = 1
mega_batch_size = 1
no_of_emotions = 6
# vocal_seq_len = 150
# vision_seq_len = 45
use_CUDA = True
use_pretrained =  True
num_workers = 20

test_mode = False
val_mode = False
train_mode = True

no_of_epochs = 1000
vocal_input_size = 74 # Dont Change
vision_input_size = 35 # Dont Change
vocal_num_layers = 2
vision_num_layers = 2
vocal_hidden_size = 1024
vision_hidden_size = 1024
dan_hidden_size = 2048
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = VocalNet(vocal_input_size, vocal_hidden_size, vocal_num_layers)
Vision_encoder = VisionNet(vision_input_size, vision_hidden_size, vision_num_layers)
Attention = DualAttention(no_of_emotions,dan_hidden_size)
Predictor = predictor(no_of_emotions,dan_hidden_size)
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
Attention = Attention.cuda()
Vision_encoder = Vision_encoder.cuda()
Predictor = Predictor.cuda()
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.MSELoss(size_average = False)
params =  list(Vocal_encoder.parameters())+ list(Attention.parameters()) +list(Vision_encoder.parameters()) + list(Predictor.parameters())
print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, lr = 0.0001)
# optimizer = torch.optim.SGD(params, lr =0.001,momentum = 0.9 )

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='attention_net'):
	filename = filename +'_'+str(state['epoch'])+'.pth.tar'
	os.system("mkdir -p DAN") 
	torch.save(state, './DAN/'+filename)
	if is_final:
		shutil.copyfile(filename, 'model_final.pth.tar')


'-------------------------------------------Setting into train mode----------------------------------------------------'

if not train_mode:
	Vision_encoder.train(False)
	Vocal_encoder.train(False)
	Attention.train(False)
	Predictor.train(False)
else:
	Vision_encoder.train(True)
	Vocal_encoder.train(True)
	Attention.train(True)
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
		# pretrained_file = './DAN/dual_attention_net_iter_8000_0.pth.tar'
		pretrained_file = './DAN/dual_attention_net__4.pth.tar'

		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		Vision_encoder.load_state_dict(checkpoint['Vision_encoder'])
		Attention.load_state_dict(checkpoint['Attention'])
		Predictor.load_state_dict(checkpoint['Predictor'])
		use_pretrained = False
		if train_mode:
			epoch = checkpoint['epoch']+1
			optimizer.load_state_dict(checkpoint['optimizer'])

	K = 0
	for i,(vision,vocal,emb,gt) in enumerate(data_loader):
		if use_CUDA:
			vision = Variable(vision.float()).cuda()
			vocal = Variable(vocal.float()).cuda()
			gt = Variable(gt.float()).cuda()

		vision_output = Vision_encoder(vision)
		vocal_output = Vocal_encoder(vocal)
		output = Attention(vocal_output,vision_output)
		outputs = Predictor(output)
		outputs = torch.clamp(outputs,0,3)
		loss = criterion(outputs, gt)
		if train_mode and K%mega_batch_size==0:
			loss.backward()
			optimizer.step()
		optimizer.zero_grad()
		Vocal_encoder.zero_grad()
		Vision_encoder.zero_grad()
		Attention.zero_grad()
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
					'Vision_encoder' : 	Vision_encoder.state_dict(),
					'Attention' : Attention.state_dict(),
					'Predictor' : Predictor.state_dict(),
					'optimizer': optimizer.state_dict(),
				}, False,'dual_attention_net_iter_'+str(K))
	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'loss' : running_loss,
			'correct' : running_corrects,
			'j_start' : 0,
			'Vocal_encoder': Vocal_encoder.state_dict(),
			'Vision_encoder' : 	Vision_encoder.state_dict(),
			'Attention' : Attention.state_dict(),
			'Predictor' : Predictor.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, False,'dual_attention_net_')
	epoch+= 1 
'------------------------------------------------------Saving model after training completion--------------------------'
if train_mode:
	save_checkpoint({
		'epoch': epoch,
		'loss' : running_loss,
		'j_start' : 0,
		'Vocal_encoder': Vocal_encoder.state_dict(),
		'Vision_encoder' : 	Vision_encoder.state_dict(),
		'Attention' : Attention.state_dict(),
		'Predictor' : Predictor.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, False)

# print('Accuracy:', accuracy_score(y_true, y_pred))
# print('F1 score:', f1_score(y_true, y_pred,average = 'weighted'))
# print('Recall:', recall_score(y_true, y_pred,average ='weighted'))
# print('Precision:', precision_score(y_true, y_pred,average = 'weighted'))
