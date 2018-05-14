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
EMBEDDING_SIZE=300

mosei=Dataloader('')
data=mosei.embeddings()



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
os.system("mkdir -p text_files_segbased")
os.system("mkdir -p text_files_segbased/train")
os.system("mkdir -p text_files_segbased/val")
os.system("mkdir -p text_files_segbased/test")
os.system("mkdir -p text_files_videobased")
os.system("mkdir -p text_files_videobased/train")
os.system("mkdir -p text_files_videobased/val")
os.system("mkdir -p text_files_videobased/test")
# pp.pprint(data)
for key,value in data.items():
	for key2,value2 in value.items():
		if key2 in train_data: 
			folder_location = "./text_files_segbased/train/" 
			videobased_folder_location = "./text_files_videobased/train/"
		elif key2 in valid_data: 
			folder_location = "./text_files_segbased/val/"
			videobased_folder_location = "./text_files_videobased/val/"
		else: 
			folder_location = "./text_files_segbased/test/"
			videobased_folder_location = "./text_files_videobased/test/"
		video_name = key2
		
		# initialise video-based embedding file name
		videobased_pickle_file = videobased_folder_location + video_name + '.pkl'		
		videobased_embeddings_features = np.empty([0,EMBEDDING_SIZE]) 
	
		# for key3,value3 in value2.items():
		key3seq = list(map(str,sorted(map(int,value2.keys()))))
		for key3 in key3seq:
			segment_id  = key3
			value3 = value2[key3]
			pickle_file = folder_location + video_name + '_' + segment_id + '.pkl'
			print(pickle_file)
		
			for idx,frame in enumerate(value3):
				videobased_embeddings_features = np.vstack((videobased_embeddings_features,frame[2]))
				if idx == 0:
					embeddings_features = frame[2]
				else:
					embeddings_features = np.vstack((embeddings_features,frame[2]))
			pickle.dump(embeddings_features, open(pickle_file,"wb"))

		# save to videobased file
		pickle.dump(videobased_embeddings_features, open(videobased_pickle_file,"wb"))

		# replicate vidoe-based file by softlinks, the order by which soft links are generated does not matter
		for key3,value3 in value2.items():
			segment_id = key3
			target_pickle_file = videobased_folder_location + video_name + '_' + segment_id + '.pkl'
			os.symlink(os.path.basename(videobased_pickle_file),target_pickle_file)

	# 		break
	# 	break
	# break


