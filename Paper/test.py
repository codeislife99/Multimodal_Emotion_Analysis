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
import sys
from torchvision import datasets, models, transforms


preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'----------------------------------------------------Resnet------------------------------------------------------------'
resnet18_filename = 'resnet_only_BEST.pth.tar'
resnet = models.resnet18(pretrained=False)  # Define resnet18 model
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 6).cuda()
# state = torch.load(resnet18_filename)
# resnet.load_state_dict(state['model'])
modules = list(resnet.children())[:-1]      # delete the last fc layer.
resnet = nn.Sequential(*modules)
# print(resnet)
'---------------------------------------------------LSTM VocalNet-------------------------------------------------------'

class VocalNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,no_of_emotions):
		super(VocalNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)
		self.linear = nn.Linear(hidden_size, no_of_emotions)


	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		# print(hiddens[-1].size()) 
		# outputs = self.linear(hiddens[-1])
		return hiddens[-1]


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

'------------------------------------------------------Hyperparameters-------------------------------------------------'
batch_size = 1
no_of_emotions = 6
seq_len = 150
use_CUDA = True
use_pretrained = False
test_mode = False
val_mode = True
train_mode = False
no_of_epochs = 1000
writer = SummaryWriter('./logs_CREMAD_resnet_vocalnet_attention')
read_prev_list = False and not test_mode
input_size = 74
num_layers = 2
hidden_size = 256
att_input_size = 512
att_hidden_size = 512
att_num_layers = 2
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = VocalNet(input_size, hidden_size,num_layers,no_of_emotions)
Attention = GatedAttention(att_input_size,att_hidden_size,att_num_layers,no_of_emotions)
curr_epoch = 0
total = 0
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = Vocal_encoder.cuda()
Attention = Attention.cuda()
resnet = resnet.cuda()
pretrained_vocal = 'LSTMvocalnet_BEST.pth.tar'
checkpoint = torch.load(pretrained_vocal)
# print(Vocal_encoder)
# Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
# modules = list(Vocal_encoder.children())[:]      # delete the last fc layer.
# Vocal_encoder = nn.Sequential(*modules)
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.CrossEntropyLoss()
params =  list(Vocal_encoder.parameters()) + list(resnet.parameters()) + list(Attention.parameters())
print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, 0.0001)

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='attention_net'):
	filename = filename +'_'+str(state['epoch'])+'.pth.tar' 
	torch.save(state, filename)
	if is_final:
		shutil.copyfile(filename, 'model_final.pth.tar')


'-------------------------------------------Setting into train mode----------------------------------------------------'

if not train_mode:
	resnet.train(False)
	Vocal_encoder.train(False)
	Attention.train(False)
else:
	resnet.train(True)
	Vocal_encoder.train(True)
	Attention.train(True)
'----------------------------------------------------------------------------------------------------------------------'
if train_mode:
	vocal_directory = "./WAVFiles/train"
	visual_directory = "../train_30"
elif val_mode:
	vocal_directory = "./WAVFiles/val"
	visual_directory = "../val_30"
else:
	vocal_directory = "./WAVFiles/test"
	visual_directory = "../test_30"
'----------------------------------------------------------------------------------------------------------------------'
names = {}
start_time = time.time()
emotions = ['A','D','F','H','N','S']
for emotion in emotions:
	for idx,filename in enumerate(sorted(glob.iglob(visual_directory+'/'+emotion+'/'+'*.jpg'))):
		if(filename[-6].isdigit()):
			start_len = len(visual_directory+'/'+emotion+'/')
			string = filename[:start_len+15]
			names[string] = max(1,int(filename[-6:-4]))
			if(names[string]<45):
				del(names[string])
time_elapsed = time.time() - start_time
print(len(names))
print(time_elapsed)

prev_loss = 0

sequences = []

emo_dict = {"ANG":0, "DIS":1 ,"FEA" :2 ,"HAP" :3, "NEU" : 4, "SAD":5}

# for filename in glob.iglob(vocal_directory +'/'+'*.mat'):
# 	struct = sio.loadmat(filename)
# 	try:
# 		sequences.append((struct['features'],emo_dict[filename[-10:-7]]))
# 	except:
# 		print(filename)

if not train_mode:
	no_of_epochs = 1
	batch_size = 1
forbidden = ['./WAVFiles/train/1050_ITS_DIS_XX.mat']
for epoch in range(curr_epoch,no_of_epochs):
	j_start = 0
	running_loss = 0
	running_corrects = 0
	if use_pretrained:
		pretrained_file = 'attention_net_iter_6500_1.pth.tar'
		# pretrained_file = 'attention_net__0.pth.tar'

		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		resnet.load_state_dict(checkpoint['resnet'])
		Attention.load_state_dict(checkpoint['Attention'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		if train_mode:
			epoch += 1
			use_pretrained = False
	# 	if train_mode:
	# 		j_start = checkpoint['j_start']
	# 		running_loss = checkpoint['loss']
	# 		running_corrects = checkpoint['correct']
	# 		curr_epoch = checkpoint['epoch']
	# 		use_pretrained = False
	K = 0
	for idx,name in enumerate(names):
		vocal_seq_input = np.empty((batch_size,seq_len,input_size), dtype = np.float32)
		target_numpy = np.empty((batch_size,) ,dtype = np.int64)

		vocal_mat_file = vocal_directory+'/'+str(name[-15:])+'.mat'  # Accessing Vocal Mat File
		if(vocal_mat_file not in forbidden):
			struct = sio.loadmat(vocal_mat_file)
		else:
			continue
		for batch in range(batch_size):
			if(emo_dict[name[-6:-3]] == emo_dict[vocal_mat_file[-10:-7]]):
				target_numpy[batch] = emo_dict[vocal_mat_file[-10:-7]]
				vocal_seq_input[batch] = struct['features']
			else:
				print(name)
				print(vocal_mat_file)
				sys.exit("Emotions don't match :( ")

		max_frames = names[name]
		mid_frame = int(math.ceil(max_frames/2.0))
		start_frame = mid_frame - 22
		end_frame = mid_frame + 22
		# print(start_frame)
		# print(end_frame)
		for frame in range(start_frame,end_frame+1):
			visual_file = name + '_frame_' + str(frame)+'.jpg'
			img =  Image.open(visual_file)
			pixels = preprocess(img)
			pixels = pixels.unsqueeze(0)
			input = Variable(pixels).cuda()
			if frame == start_frame:
				image_batch = input
			else:
				image_batch = torch.cat((image_batch, input), 0)

		resnet_output = resnet(image_batch)
		# print(resnet_output.size())
		target = Variable(torch.from_numpy(target_numpy)).cuda()  
		vocal_seq_i = Variable(torch.from_numpy(vocal_seq_input)).cuda()
		vocal_output = Vocal_encoder(vocal_seq_i)

		# print(vocal_output.size())
		# print(vocal_mat_file)
		# print(name + '_frame_' + str(start_frame)+'.jpg')
		outputs = Attention(vocal_output,resnet_output)

		loss = criterion(outputs, target)

		optimizer.zero_grad()
		Vocal_encoder.zero_grad()
		resnet.zero_grad()
		Attention.zero_grad()

		if train_mode:
			loss.backward()
			optimizer.step()


		_, preds = torch.max(outputs.data, 1)
		running_loss += loss.data[0]
		running_corrects += torch.sum(preds == target.data)   

		running_accuracy = 100.0*float(running_corrects)/float(K+batch_size)
		K+=1
		average_loss = float(running_loss)/float(K)

		print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f, Accuracy: %.4f'
		% (epoch+1, K+batch_size, average_loss, running_accuracy))
		if (K+batch_size)%250==0:
			save_checkpoint({
				'epoch': epoch,
				'accuracy': running_accuracy,
				'loss' : running_loss,
				'correct' : running_corrects,
				'j_start' : 0,
				'Vocal_encoder': Vocal_encoder.state_dict(),
				'resnet' : resnet.state_dict(),
				'Attention' : Attention.state_dict(),
				'optimizer': optimizer.state_dict(),
			}, False,'attention_net_iter_'+str(K+batch_size))			
	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'accuracy': running_accuracy,
			'loss' : running_loss,
			'correct' : running_corrects,
			'j_start' : 0,
			'Vocal_encoder': Vocal_encoder.state_dict(),
			'resnet' : resnet.state_dict(),
			'Attention' : Attention.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, False,'attention_net_')
'------------------------------------------------------Saving model after training completion--------------------------'
if train_mode:
	save_checkpoint({
		'epoch': epoch,
		'accuracy': running_accuracy,
		'loss' : running_loss,
		'correct' : running_corrects,
		'j_start' : 0,
		'Vocal_encoder': Vocal_encoder.state_dict(),
		'resnet' : resnet.state_dict(),
		'Attention' : Attention.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, False)