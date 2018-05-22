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
from torch.nn.parameter import Parameter
from models.highway import GatedMemUpdate

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)

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

'---------------------------------------------------LSTM TextNet-------------------------------------------------------'

class WordvecNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers):
		super(WordvecNet, self).__init__()
		self.rnn = nn.LSTM(input_size,hidden_size,num_layers,bidirectional = True)

	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.rnn(x)
		return hiddens[-1]



'---------------------------------------------------LSTM VisualNet-------------------------------------------------------'

class VisionNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers):
		super(VisionNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)


	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		return hiddens[-1]

'---------------------------------------------------Memory to Emotion Decoder------------------------------------------'
class predictor_v(nn.Module):
	def __init__(self,no_of_emotions,hidden_size,output_scale_factor = 1, output_shift = 0):
		super(predictor_v, self).__init__()
		self.fc = nn.Linear(hidden_size, no_of_emotions)


	def forward(self,x):
		x = self.fc(x)

		return x

class predictor_a(nn.Module):
	def __init__(self,no_of_emotions,hidden_size):
		super(predictor_a, self).__init__()
		self.fc = nn.Linear(hidden_size, no_of_emotions)
	def forward(self,x):
		x = self.fc(x)


		return x
class predictor_t(nn.Module):
	def __init__(self,no_of_emotions,hidden_size,output_scale_factor = 1, output_shift = 0):
		super(predictor_t, self).__init__()
		self.fc = nn.Linear(hidden_size, no_of_emotions)

	def forward(self,x):
		x = self.fc(x)

		return x

class late_weighting(nn.Module):
	def __init__(self):
		super(late_weighting, self).__init__()
		self.v = Parameter(torch.FloatTensor([0,0,0,0,0,0]))
		self.a = Parameter(torch.FloatTensor([0,0,0,0,0,0]))
		self.t = Parameter(torch.FloatTensor([0,0,0,0,0,0]))

	def forward(self,v,a,t):
		x = self.v*v+self.a*a+self.t*t
		return x

'------------------------------------------------------Hyperparameters-------------------------------------------------'
batch_size = 1
mega_batch_size = 1
no_of_emotions = 6
use_CUDA = True
use_pretrained = True
use_pretrained_encoders = False
num_workers = 20

test_mode = True
val_mode = False
train_mode = False

no_of_epochs = 30
vocal_input_size = 74 # Dont Change
vision_input_size = 35 # Dont Change
wordvec_input_size = 300
vocal_num_layers = 2
vision_num_layers = 2
wordvec_num_layers = 2
vocal_hidden_size = 512
vision_hidden_size = 512
wordvec_hidden_size = 512
dan_hidden_size = 1024
gated_mem = False
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = VocalNet(vocal_input_size, vocal_hidden_size, vocal_num_layers)
Vision_encoder = VisionNet(vision_input_size, vision_hidden_size, vision_num_layers)
Wordvec_encoder = WordvecNet(wordvec_input_size, wordvec_hidden_size, wordvec_num_layers)
PredictorV = predictor_v(no_of_emotions,dan_hidden_size)
PredictorA = predictor_a(no_of_emotions,dan_hidden_size)
PredictorT = predictor_t(no_of_emotions,dan_hidden_size)
Lateweight = late_weighting()

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
Vision_encoder = Vision_encoder.cuda()
Wordvec_encoder = Wordvec_encoder.cuda()
PredictorV = PredictorV.cuda()
PredictorA = PredictorA.cuda()
PredictorT = PredictorT.cuda()
Lateweight = Lateweight.cuda()
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.MSELoss(size_average = False)
params =  list(Vocal_encoder.parameters()) + list(Wordvec_encoder.parameters()) + list(Vision_encoder.parameters()) + list(PredictorV.parameters()) + list(PredictorA.parameters()) + list(PredictorT.parameters()) + list(Lateweight.parameters())
print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, lr = 0.0001)
# optimizer = torch.optim.SGD(params, lr =0.001,momentum = 0.9 )

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='late_weight_new'):
	filename = filename +'_'+str(state['epoch'])+'.pth.tar'
	os.system("mkdir -p late_weight") 
	torch.save(state, './late_weight/'+filename)
	if is_final:
		shutil.copyfile(filename, 'model_final.pth.tar')


'-------------------------------------------Setting into train mode----------------------------------------------------'

if not train_mode:
	Vision_encoder.train(False)
	Vocal_encoder.train(False)
	Wordvec_encoder.train(False)
	PredictorV.train(False)
	PredictorA.train(False)
	PredictorT.train(False)
	Lateweight.train(False)
else:
	Vision_encoder.train(True)
	Vocal_encoder.train(True)
	Wordvec_encoder.train(True)
	PredictorV.train(True)
	PredictorA.train(True)
	PredictorT.train(True)
	Lateweight.train(True)
'----------------------------------------------------------------------------------------------------------------------'
epoch = 0
y_true = []
y_pred = []
while epoch<no_of_epochs:
	j_start = 0
	running_loss = 0
	running_corrects = 0
	if use_pretrained_encoders:
		pretrained_file_v = './vision_only/vision_net__10.pth.tar'
		pretrained_file_a = './vocal_only/vocal_net__7.pth.tar'
		pretrained_file_t = './verbal_only/verbal_net__4.pth.tar'

		checkpoint_v = torch.load(pretrained_file_v) 
		checkpoint_a = torch.load(pretrained_file_a) 
		checkpoint_t = torch.load(pretrained_file_t) 

		Vision_encoder.load_state_dict(checkpoint_v['Vision_encoder'])
		PredictorV.load_state_dict(checkpoint_v['Predictor'])

		Vocal_encoder.load_state_dict(checkpoint_a['Vocal_encoder'])
		PredictorA.load_state_dict(checkpoint_a['Predictor'])

		Wordvec_encoder.load_state_dict(checkpoint_t['Wordvec_encoder'])
		PredictorT.load_state_dict(checkpoint_t['Predictor'])

		use_pretrained_encoders = False

	if use_pretrained:
		pretrained_file = './late_weight/late_weight_net__0.pth.tar'

		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		Vision_encoder.load_state_dict(checkpoint['Vision_encoder'])
		Wordvec_encoder.load_state_dict(checkpoint['Wordvec_encoder'])
		PredictorV.load_state_dict(checkpoint['PredictorV'])
		PredictorA.load_state_dict(checkpoint['PredictorA'])
		PredictorT.load_state_dict(checkpoint['PredictorT'])
		Lateweight.load_state_dict(checkpoint['Lateweight'])

		use_pretrained = False
		if train_mode:
			epoch = checkpoint['epoch']+1
			optimizer.load_state_dict(checkpoint['optimizer'])

	K = 0
	for i,(vision,vocal,emb,gt) in enumerate(data_loader):
		if use_CUDA:

			vision = Variable(vision.float()).cuda()
			vocal = Variable(vocal.float()).cuda()
			emb = Variable(emb.float()).cuda()
			gt = Variable(gt.float()).cuda()

		vision_output = Vision_encoder(vision)
		vocal_output = Vocal_encoder(vocal)
		emb_output = Wordvec_encoder(emb)
		outputs_v = PredictorV(vision_output)
		outputs_v = torch.clamp(outputs_v,0,3)
		outputs_a = PredictorA(vocal_output)
		outputs_a = torch.clamp(outputs_a,0,3)
		outputs_t = PredictorT(emb_output)
		outputs_t = torch.clamp(outputs_t,0,3)

		outputs = Lateweight(outputs_v,outputs_a,outputs_t)
		outputs = torch.clamp(outputs,0,3)
		loss = criterion(outputs, gt)
		if train_mode and K%mega_batch_size==0:
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			Vocal_encoder.zero_grad()
			Vision_encoder.zero_grad()
			Wordvec_encoder.zero_grad()
			PredictorV.zero_grad()
			PredictorA.zero_grad()
			PredictorT.zero_grad()
			Lateweight.zero_grad()

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
					'Vocal_encoder': Vocal_encoder.state_dict(),
					'Vision_encoder' : 	Vision_encoder.state_dict(),
					'Wordvec_encoder' : Wordvec_encoder.state_dict(),
					'PredictorV' : PredictorV.state_dict(),
					'PredictorA' : PredictorA.state_dict(),
					'PredictorT' : PredictorT.state_dict(),
					'Lateweight'  : Lateweight.state_dict(),
					'optimizer': optimizer.state_dict(),
				}, False,'late_weight_net_iter_'+str(K))
	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'loss' : running_loss,
			'correct' : running_corrects,
			'Vocal_encoder': Vocal_encoder.state_dict(),
			'Vision_encoder' : 	Vision_encoder.state_dict(),
			'Wordvec_encoder' : Wordvec_encoder.state_dict(),
			'PredictorV' : PredictorV.state_dict(),
			'PredictorA' : PredictorA.state_dict(),
			'PredictorT' : PredictorT.state_dict(),
			'Lateweight'  : Lateweight.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, False,'late_weight_net_')
	epoch+= 1 
'------------------------------------------------------Saving model after training completion--------------------------'
if train_mode:
	save_checkpoint({
		'epoch': epoch,
		'loss' : running_loss,
		'Vocal_encoder': Vocal_encoder.state_dict(),
		'Vision_encoder' : 	Vision_encoder.state_dict(),
		'Wordvec_encoder' : Wordvec_encoder.state_dict(),
		'PredictorV' : PredictorV.state_dict(),
		'PredictorA' : PredictorA.state_dict(),
		'PredictorT' : PredictorT.state_dict(),
		'Lateweight'  : Lateweight.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, False)

# print('Accuracy:', accuracy_score(y_true, y_pred))
# print('F1 score:', f1_score(y_true, y_pred,average = 'weighted'))
# print('Recall:', recall_score(y_true, y_pred,average ='weighted'))
# print('Precision:', precision_score(y_true, y_pred,average = 'weighted'))
