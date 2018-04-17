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
import csv
import pandas as pd

preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'----------------------------------------------------Resnet------------------------------------------------------------'
class resnetAU(nn.Module):
	def __init__(self,resnet_pre):
		super(resnetAU, self).__init__()
		#defining layers in convnet
		self.resnet=resnet_pre
		self.fcau = nn.Linear(512,17)

	def forward(self, x):
		x = self.resnet(x)
		x = x.squeeze(2)
		x = x.squeeze(2)
		x = self.fcau(x)
		return x

resnet18_filename = 'resnet_only_BEST.pth.tar'
resnet_pre = models.resnet18(pretrained=True)  # Define resnet18 model
num_ftrs = resnet_pre.fc.in_features
resnet_pre.fc = nn.Linear(num_ftrs, 6).cuda()
state = torch.load(resnet18_filename)
resnet_pre.load_state_dict(state['model'])
modules = list(resnet_pre.children())[:-1]      # delete the last fc layer and the avg.pool layer
resnet_pre = nn.Sequential(*modules)
resnet = resnetAU(resnet_pre)
# print(resnet)

'------------------------------------------------------Hyperparameters-------------------------------------------------'
batch_size = 8
no_of_AUs = 17
use_CUDA = True
use_pretrained = False
test_mode = False
val_mode = False
train_mode = True
no_of_epochs = 1000
writer = SummaryWriter('./logs_CREMAD_AU')
read_prev_list = False and not test_mode
'----------------------------------------------------------------------------------------------------------------------'
curr_epoch = 0
total = 0
'----------------------------------------------------------------------------------------------------------------------'
resnet = resnet.cuda()
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.MSELoss()
params =  list(resnet.parameters())
print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, 0.000001)

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='AU_net'):
	filename = filename +'_'+str(state['epoch'])+'.pth.tar' 
	torch.save(state, './AUNet/'+filename)
	if is_final:
		shutil.copyfile(filename, 'model_final.pth.tar')


'-------------------------------------------Setting into train mode----------------------------------------------------'

if not train_mode:
	resnet.train(False)
else:
	resnet.train(True)
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
all_frames = []
for emotion in emotions:
	for idx,filename in enumerate(sorted(glob.iglob(visual_directory+'/'+emotion+'/'+'*.jpg'))):
		all_frames.append(filename)
		start_len = len(visual_directory+'/'+emotion+'/')
		# if(filename[-6].isdigit()):
		# 	start_len = len(visual_directory+'/'+emotion+'/')
		# 	string = filename[:start_len+15]
		# 	names[string] = max(1,int(filename[-6:-4]))
			# if(names[string]<45):
			# 	del(names[string])
time_elapsed = time.time() - start_time
print(len(names))
print(time_elapsed)

prev_loss = 0

sequences = []

emo_dict = {"ANG":0, "DIS":1 ,"FEA" :2 ,"HAP" :3, "NEU" : 4, "SAD":5}

if not train_mode:
	no_of_epochs = 1
	batch_size = 1
all_frames = np.random.permutation(np.array(all_frames))
for epoch in range(curr_epoch,no_of_epochs):
	j_start = 0
	running_loss = 0
	running_corrects = 0
	if use_pretrained:
		pretrained_file = 'attention_net__2.pth.tar'

		checkpoint = torch.load(pretrained_file)
		resnet.load_state_dict(checkpoint['resnet'])
		if train_mode:
			epoch = checkpoint['epoch']+1
			use_pretrained = False
			optimizer.load_state_dict(checkpoint['optimizer'])

	K = 0

	for j in range(j_start,len(all_frames),batch_size):

		batch_frames = all_frames[j : j + batch_size]
		target_numpy = np.empty((batch_size,no_of_AUs), dtype = np.float32)

		for idx,name in enumerate(batch_frames):
			img  = Image.open(name)
			pixels = preprocess(img)
			pixels = pixels.unsqueeze(0)
			input = Variable(pixels).cuda()
			if idx == 0:
				image_batch = input
			else:
				image_batch = torch.cat((image_batch, input), 0)
			csv_file = name[start_len:start_len+15]+'.csv'
			if(name[-6].isdigit()):
				frame_no = int(name[-6:-4])
			else:
				frame_no = int(name[-5:-4])
			# print(name, csv_file, frame_no)

			csv_file_name = './processed/'+csv_file
			df = pd.read_csv(csv_file_name)
			# print(df)
			df = df.iloc[[frame_no-1],[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
			# print(df.values[0])
			target_numpy[idx] = df.values[0]

			# with open('./processed/'+csv_file, 'r') as csvfile:
			# 	csvr = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
			# 	csvr = list(csvr)
			# 	if len(csvr) <= frame_no:
			# 		row_no = len(csvr) - 1
			# 	else:
			# 		row_no = frame_no

			# 	print(csvr[frame_no][5:22])
			# 	target_numpy[idx] = csvr[frame_no][5:22]      # The 3rd row
			# 	print(target_numpy)


			# for batch in range(batch_size):
			# 	if(emo_dict[name[-6:-3]] == emo_dict[vocal_mat_file[-10:-7]]):
			# 		target_numpy[batch] = emo_dict[vocal_mat_file[-10:-7]]
			# 	else:
			# 		print(name)
			# 		sys.exit("Emotions don't match :( ")

			# start_frame = 0
			# end_frame = mid_frame + 22
			# # print(start_frame)
			# # print(end_frame)
			# for frame in range(start_frame,end_frame+1):
			# 	visual_file = name + '_frame_' + str(frame)+'.jpg'
			# 	img =  Image.open(visual_file)
			# 	pixels = preprocess(img)
			# 	pixels = pixels.unsqueeze(0)
			# 	input = Variable(pixels).cuda()
			# 	if frame == start_frame:
			# 		image_batch = input
			# 	else:
			# 		image_batch = torch.cat((image_batch, input), 0)
		resnet_output = resnet(image_batch)
		# print(target_numpy)

		# print(resnet_output.size())
		target = Variable(torch.from_numpy(target_numpy)).cuda()  

		# print(vocal_output.size())
		# print(vocal_mat_file)
		# print(name + '_frame_' + str(start_frame)+'.jpg')

		loss = criterion(resnet_output, target)

		optimizer.zero_grad()
		resnet.zero_grad()

		if train_mode:
			loss.backward()
			optimizer.step()


		running_loss += loss.data[0]
		K+=1
		average_loss = float(running_loss)/float(K)

		print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f'
		% (epoch+1, j+batch_size, average_loss))
		if (j+batch_size)%10000==0:
			save_checkpoint({
				'epoch': epoch,
				'loss' : running_loss,
				'j_start' : 0,
				'resnet' : resnet.state_dict(),
				'optimizer': optimizer.state_dict(),
			}, False,'AU_net_iter_'+str(j+batch_size))			
	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'loss' : running_loss,
			'j_start' : 0,
			'resnet' : resnet.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, False,'AU_net_iter_')
'------------------------------------------------------Saving model after training completion--------------------------'
if train_mode:
	save_checkpoint({
		'epoch': epoch,
		'accuracy': running_accuracy,
		'loss' : running_loss,
		'correct' : running_corrects,
		'j_start' : 0,
		'resnet' : resnet.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, False)