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
import matplotlib.cm as cm
import cv2
import pandas as pd 

preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'---------------------------------------------------LSTM VocalNet-------------------------------------------------------'

class VocalNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,no_of_emotions):
		super(VocalNet, self).__init__()
		self.lstm = nn.LSTM(74,128,num_layers,bidirectional=False)
		self.linear = nn.Linear(hidden_size, 6)


	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		return hiddens[-1]
'---------------------------------------------------Emotion Decoder----------------------------------------------------'
class predictor(nn.Module):
	def __init__(self,no_of_emotions):
		super(predictor, self).__init__()
		self.fc = nn.Linear(256, no_of_emotions)
	def forward(self,x):
		x = self.fc(x)
		return x
'------------------------------------------------------Hyperparameters-------------------------------------------------'
batch_size = 1
no_of_emotions = 6
vocal_seq_len = 150
vision_seq_len = 45
use_CUDA = True


use_pretrained_speech = True
use_pretrained_song = False
use_CREMAD = False
test_mode = True
val_mode = False
train_mode = False
speech_mode = True

if speech_mode:
	no_of_emotions = 8


no_of_epochs = 1000
writer = SummaryWriter('./logs_CREMAD_OF_vocalnet_attention')
vocal_input_size = 74 # Dont Change
vocal_num_layers = 2
vocal_hidden_size = 128
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = VocalNet(vocal_input_size, vocal_hidden_size,vocal_num_layers,no_of_emotions)
Predictor = predictor(no_of_emotions)
curr_epoch = 0
total = 0
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = Vocal_encoder.cuda()
Predictor = Predictor.cuda()
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.CrossEntropyLoss()
params =  list(Vocal_encoder.parameters())+ list(Predictor.parameters())
# params =  list(Attention.parameters()) +list(Vision_encoder.parameters()) + list(Predictor.parameters())

print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, lr =0.001)
# optimizer = torch.optim.SGD(params, lr =0.001,momentum = 0.9 )

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='audio_only_net'):
	filename = filename +'_'+str(state['epoch'])+'.pth.tar' 
	torch.save(state, './Audio_Only/'+filename)
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
if train_mode:
	vocal_directory = "../MATLAB_COVAREP/WAVFiles/train"
	visual_numbers = np.arange(5,25)
elif val_mode:
	vocal_directory = "../MATLAB_COVAREP/WAVFiles/valid"
	visual_numbers = np.array([3,4])
else:
	vocal_directory = "../MATLAB_COVAREP/WAVFiles/test"
	visual_numbers = np.array([1,2])
'----------------------------------------------------------------------------------------------------------------------'
start_time = time.time()
all_csv_files = []
for actor_no in visual_numbers:
	actor_str = str(actor_no).zfill(2)

	for file in glob.glob("../processed/01-02-*-"+actor_str+".csv"):
		all_csv_files.append(file)
time_elapsed = time.time() - start_time
print("Number of Files = ", len(all_csv_files))
print(time_elapsed)

prev_loss = 0

sequences = []

emo_dict = {1 :"neutral", 2 :"calm", 3 :"happy", 4 : "sad", 5 :"angry", 6 : "fearful", 7 : "disgust", 8 : "surprised"}
# for filename in glob.iglob(vocal_directory +'/'+'*.mat'):
# 	struct = sio.loadmat(filename)
# 	try:
# 		sequences.append((struct['features'],emo_dict[filename[-10:-7]]))
# 	except:
# 		print(filename)

if not train_mode:
	no_of_epochs = 1
	batch_size = 1
epoch = 0

all_csv_files = np.random.permutation(np.array(all_csv_files))
while epoch<no_of_epochs:
	j_start = 0
	running_loss = 0
	running_corrects = 0
	if use_CREMAD:
		pretrained_file = 'OF_attention_net__5.pth.tar'
		# pretrained_file = 'attention_net__0.pth.tar'

		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		use_CREMAD = False

	elif use_pretrained_speech:
		pretrained_file = './Audio_Only/audio_only_net__17.pth.tar'
		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		if speech_mode:
			Predictor.load_state_dict(checkpoint['Predictor'])
		if train_mode and speech_mode:
			epoch = checkpoint['epoch']+1
			use_pretrained_speech = False
			optimizer.load_state_dict(checkpoint['optimizer'])

	elif use_pretrained_song:
		pretrained_file = './Audio_Only_SONG/audio_only_net__0.pth.tar'
		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		Predictor.load_state_dict(checkpoint['Predictor'])
		if train_mode and not speech_mode:
			epoch = checkpoint['epoch']+1
			use_pretrained_song = False
			optimizer.load_state_dict(checkpoint['optimizer'])

	K = 0
	# all_csv_files = ["../processed/01-01-02-02-02-01-05.csv"]
	# all_csv_files = ["../processed/01-01-05-01-01-02-06.csv"]

	for idx,csv_file_path in enumerate(all_csv_files):
		# print(csv_file_path)
		name = csv_file_path[13:-4]
		vocal_seq_input0 = np.empty((1,vocal_seq_len,vocal_input_size), dtype = np.float32)
		vocal_seq_input1 = np.empty((1,vocal_seq_len,vocal_input_size), dtype = np.float32)

		target_numpy = np.empty((1,) ,dtype = np.int64)

		vocal_mat_file = vocal_directory+'/'+"03"+str(name[2:])+'.mat'  # Accessing Vocal Mat File
		struct = sio.loadmat(vocal_mat_file)

		# print(struct)
		no_of_seconds = struct['features'].shape[0]*0.01
		no_of_frames = no_of_seconds*29.97
		if no_of_frames < 90:
			print(csv_file_path)
			continue

		struct['features'][[struct['features']<(-1e8)]] = 0


		target_numpy[0] = int(name[6:8])-1
		yolo = Variable(torch.from_numpy(target_numpy)).cuda()  

		for iteration in range(2):
			if iteration == 0:
				vocal_seq_input0[0] = struct['features'][0:150]
				vocal_seq_input1[0] = struct['features'][150:300]
			else:
				vocal_seq_input0[0] = struct['features'][-300:-150]
				vocal_seq_input1[0] = struct['features'][-150:]

			# print(resnet_output0)
			target = Variable(torch.from_numpy(target_numpy)).cuda()  
			vocal_seq_i0 = Variable(torch.from_numpy(vocal_seq_input0)).cuda()
			vocal_seq_i1 = Variable(torch.from_numpy(vocal_seq_input1)).cuda()
			vocal_output0 = Vocal_encoder(vocal_seq_i0)
			vocal_output1 = Vocal_encoder(vocal_seq_i1)

		
			concat_output = torch.cat((vocal_output0,vocal_output1),dim=1)
			# print(mem
			outputs = Predictor(concat_output)
			# print(outputs.size())
			loss = criterion(outputs, target)
			# print(outputs)
			# print(target)
			optimizer.zero_grad()
			Vocal_encoder.zero_grad()
			Predictor.zero_grad()

			running_loss += loss.data[0]
			if train_mode:
				loss.backward()
				optimizer.step()
				_, preds = torch.max(outputs.data, 1)
				# print(target.data)
				running_corrects += torch.sum(preds == target.data)   
				K+=1
				running_accuracy = 100.0*float(running_corrects)/float(K)
				
				average_loss = float(running_loss)/float(K)
				print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f, Accuracy: %.4f'
				% (epoch+1, K, average_loss, running_accuracy))
				if (K+batch_size)%250==0:
					save_checkpoint({
						'epoch': epoch,
						'accuracy': running_accuracy,
						'loss' : running_loss,
						'correct' : running_corrects,
						'j_start' : 0,
						'Vocal_encoder': Vocal_encoder.state_dict(),
						'Predictor' : Predictor.state_dict(),
						'optimizer': optimizer.state_dict(),
					}, False,'audio_only_net_iter_'+str(K+batch_size))

			else:
				if iteration == 0:
					avg_softmax_outputs1 = F.softmax(outputs,dim=-1)
					# print(avg_softmax_outputs1)
				else:
					avg_softmax_outputs2 = F.softmax(outputs,dim=-1)
					# print(avg_softmax_outputs2)

		if not train_mode:
			avg_all = (avg_softmax_outputs1+avg_softmax_outputs2)
			# print(avg_all)
			_, preds = torch.max(avg_all.data, 1)
			running_corrects += torch.sum(preds == yolo.data)   
			K+=1
			running_accuracy = 100.0*float(running_corrects)/float(K)
			average_loss = float(running_loss)/float(2*K)
			if val_mode:
				print('Validating -- Epoch [%d], Sample [%d], Average Loss: %.4f, Accuracy: %.4f'
				% (epoch+1, K, average_loss, running_accuracy))
			else:
				print('Testing -- Epoch [%d], Sample [%d], Average Loss: %.4f, Accuracy: %.4f'
				% (epoch+1, K, average_loss, running_accuracy))				
			# print("HERE")

	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'accuracy': running_accuracy,
			'loss' : running_loss,
			'correct' : running_corrects,
			'j_start' : 0,
			'Vocal_encoder': Vocal_encoder.state_dict(),
			'Predictor' : Predictor.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, False,'audio_only_net_')
	epoch+= 1 
'------------------------------------------------------Saving model after training completion--------------------------'
if train_mode:
	save_checkpoint({
		'epoch': epoch,
		'accuracy': running_accuracy,
		'loss' : running_loss,
		'correct' : running_corrects,
		'j_start' : 0,
		'Vocal_encoder': Vocal_encoder.state_dict(),
		'Predictor' : Predictor.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, False)