import cPickle as pickle
import pprint
import random
import os
import numpy as np

with open('covarep.pkl', 'rb') as f:
	data = pickle.load(f)
pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(data[0])
# pp.pprint(len(data))
with open('train.pkl', 'rb') as f:
	train_data = pickle.load(f)
with open('valid.pkl', 'rb') as f:
	valid_data = pickle.load(f)
with open('test.pkl', 'rb') as f:
	test_data = pickle.load(f)


for key,value in data.items():
	pp.pprint(key)
	for key2,value2 in value.items():
		if key2 in train_data: 
			folder_location = "./Multimodal_Emotion_Analysis/audio_files/train/" 
		elif key2 in valid_data: 
			folder_location = "./Multimodal_Emotion_Analysis/audio_files/val/"
		else: 
			folder_location = "./Multimodal_Emotion_Analysis/audio_files/test/"
		video_name = key2
		pp.pprint(key2)
		for key3,value3 in value2.items():
			segment_id  = key3
			pickle_file = folder_location + video_name + '_' + segment_id + '.pkl'
			print(pickle_file)
			pp.pprint(value3)
			# for idx,frame in enumerate(value3):
			# 	if idx == 0:
			# 		facet_features = frame[2]
			# 	else:
			# 		facet_features = np.vstack((facet_features,frame[2]))
			# # pp.pprint(facet_features)
			pickle.dump(facet_features, open(pickle_file, "wb" ))
			# break
		# break
	# break


