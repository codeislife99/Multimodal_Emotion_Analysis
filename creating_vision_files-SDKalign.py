# import cPickle as pickle
from mmdata import Dataloader, Dataset

import pickle
import pprint
import random
import os
import numpy as np

mosei=Dataloader('')
mosei_facet=mosei.facet()
mosei_covarep=mosei.covarep()
mosei_av=Dataset.merge(mosei_facet,mosei_covarep)
data=mosei_av.align('covarep')

train_data=mosei.train()
valid_data=mosei.valid()
test_data=mosei.test()

os.system("mkdir -p vision_files_align")
os.system("mkdir -p vision_files_align/train")
os.system("mkdir -p vision_files_align/val")
os.system("mkdir -p vision_files_align/test")


# for key,value in data.items():
for key,value in data.items():
	if key != 'facet':
		print('I am skipping '+key)
		continue	
	else:
		print('Start dumping '+key)

	for key2,value2 in value.items():
		if key2 in train_data: 
			folder_location = "./vision_files_align/train/" 
		elif key2 in valid_data: 
			folder_location = "./vision_files_align/val/"
		else: 
			folder_location = "./vision_files_align/test/"
		video_name = key2
		for key3,value3 in value2.items():
			segment_id  = key3
			pickle_file = folder_location + video_name + '_' + segment_id + '.pkl'
			print(pickle_file)
			# print(value3[0][2])
			for idx,frame in enumerate(value3):
				if idx == 0:
					facet_features = frame[2]
				else:
					facet_features = np.vstack((facet_features,frame[2]))
			# pp.pprint(facet_features)
			pickle.dump(facet_features, open(pickle_file, "wb" ))
			# break
		# break
	# break


