# import cPickle as pickle
from mmdata import Dataloader
import pickle
import pprint
import random
import os
import numpy as np
# import ipdb

# with open('covarep.pkl', 'rb') as f:
#	# data = pickle.load(f)
#	data = pickle.load(f,encoding='bytes')
mosei=Dataloader('')
data=mosei.covarep()

# ipdb.set_trace()
pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(data[0])
# pp.pprint(len(data))
train_data=mosei.train()
valid_data=mosei.valid()
test_data=mosei.test()

# with open('train.pkl', 'rb') as f:
# 	# train_data = pickle.load(f)
# 	train_data = pickle.load(f,encoding='bytes')
# with open('valid.pkl', 'rb') as f:
# 	# valid_data = pickle.load(f)
# 	valid_data = pickle.load(f,encoding='bytes')
# with open('test.pkl', 'rb') as f:
# 	# test_data = pickle.load(f)
# 	test_data = pickle.load(f,encoding='bytes')
os.system("mkdir -p audio_files")
os.system("mkdir -p audio_files/train")
os.system("mkdir -p audio_files/val")
os.system("mkdir -p audio_files/test")
# pp.pprint(data)
for key,value in data.items():
# for key,value in data.iteritems():
	# pp.pprint(key)
	for key2,value2 in value.items():
		if key2 in train_data: 
			folder_location = "./audio_files/train/" 
		elif key2 in valid_data: 
			folder_location = "./audio_files/val/"
		else: 
			folder_location = "./audio_files/test/"
		video_name = key2
		# pp.pprint(key2)
		for key3,value3 in value2.items():
			segment_id  = key3
			pickle_file = folder_location + video_name + '_' + segment_id + '.pkl'
			print(pickle_file)
			# pp.pprint(value3)
			for idx,frame in enumerate(value3):
				if idx == 0:
					covarep_features = frame[2]
				else:
					covarep_features = np.vstack((covarep_features,frame[2]))
			# pp.pprint(facet_features)
			pickle.dump(covarep_features, open(pickle_file,"wb"))
	# 		break
	# 	break
	# break


