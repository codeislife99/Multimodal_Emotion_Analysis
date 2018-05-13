import cPickle as pickle
import os
import glob
import sys

os.system("mkdir -p audio_files_74")
splits = ['train','test','val']
for split in splits:
	os.system("mkdir -p audio_files_74/"+split)
	path = './audio_files/'+split+'/*.pkl'
	count = 0
	for file in glob.glob(path):
		with open(file,'rb') as f:
			if sys.version_info[0]==2:
				data = pickle.load(f)
			else:
				data = pickle.load(f,encoding = 'latin1')
			# print(data.shape[1])
			if data.shape[1] == 74:
				os.system("mv "+file+" ./audio_files_74/"+split+"/")
				# print("mv "+file+" ./audio_files_74/"+split+"/")
				count += 1
	print(split, count)