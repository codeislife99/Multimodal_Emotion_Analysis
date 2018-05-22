# import cPickle as pickle
from mmdata import Dataloader, Dataset
import pickle
import pprint
import random
import os
import numpy as np
# import ipdb

# with open('covarep.pkl', 'rb') as f:
#	# data = pickle.load(f)
#	data = pickle.load(f,encoding='bytes')
EMBEDDING_SIZE=300

mosei=Dataloader('')
mosei_embedding=mosei.embeddings()
mosei_covarep=mosei.covarep()
mosei_at=Dataset.merge(mosei_embedding,mosei_covarep)

data=mosei_at.align('covarep')



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
os.system("mkdir -p text_files_segbased_align")
os.system("mkdir -p text_files_segbased_align/train")
os.system("mkdir -p text_files_segbased_align/val")
os.system("mkdir -p text_files_segbased_align/test")
# os.system("mkdir -p text_files_videobased_align")
# os.system("mkdir -p text_files_videobased_align/train")
# os.system("mkdir -p text_files_videobased_align/val")
# os.system("mkdir -p text_files_videobased_align/test")
# pp.pprint(data)
for key,value in data.items():
	if key != 'embeddings':
		print ('I am skipping '+key)
		continue
	else:
		print ('Start dumping '+key)

	for key2,value2 in value.items():
		if key2 in train_data: 
			folder_location = "./text_files_segbased_align/train/" 
		elif key2 in valid_data: 
			folder_location = "./text_files_segbased_align/val/"
		else: 
			folder_location = "./text_files_segbased_align/test/"
		video_name = key2
		
		key3seq = list(map(str,sorted(map(int,value2.keys()))))
		for key3 in key3seq:
			segment_id  = key3
			value3 = value2[key3]
			pickle_file = folder_location + video_name + '_' + segment_id + '.pkl'
			print(pickle_file)
		
			for idx,frame in enumerate(value3):
				if idx == 0:
					embeddings_features = frame[2]
				else:
					embeddings_features = np.vstack((embeddings_features,frame[2]))
			if embeddings_features.shape[0] == 300:
				embeddings_features = np.reshape(embeddings_features, [-1, 300])
			pickle.dump(embeddings_features, open(pickle_file,"wb"))

	# 		break
	# 	break
	# break


