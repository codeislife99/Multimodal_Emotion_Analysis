import cPickle as pickle
import pprint
import random
import os
import numpy as np

with open('emotions.pkl', 'rb') as f:
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

# pp.pprint(data)
# for key,value in data.items():
# 	pp.pprint(key)
train_set = set([])
val_set = set([])
test_set = set([])

for key2,value2 in data.items():
	if key2 in train_data: 
		folder_location = "./gt_emotions_files/train/" 
		train_set.add(key2)
	elif key2 in valid_data: 
		folder_location = "./gt_emotions_files/val/"
		val_set.add(key2)
	else: 
		folder_location = "./gt_emotions_files/test/"
	# video_name = key2
	# for key3,value3 in value2.items():
	# 	segment_id  = key3
	# 	pickle_file = folder_location + video_name + '_' + segment_id + '.pkl'
	# 	print(pickle_file)
	# 	# pp.pprint(facet_features)
	# 	pickle.dump(value3, open(pickle_file, "wb" ))
		# break
	# break
# break

print(len(train_set))
print(len(val_set))
print(len(train_data))
print(len(valid_data))
