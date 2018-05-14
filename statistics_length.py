import cPickle as pickle
import os
import glob
import subprocess
import json

from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import time


start_time = time.time()


def get_len(filename):
   result = subprocess.Popen(["ffprobe", filename, '-print_format', 'json', '-show_streams', '-loglevel', 'quiet'],
     stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
   return float(json.loads(result.stdout.read())['streams'][0]['duration'])


splits = ['train','val', 'test']
bins = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", "20+"]
emo_bins = [{"0-2":0,"2-4":0,"4-6":0,"6-8":0,"8-10":0, "10-15":0, "15-20":0, "20+":0},
			{"0-2":0,"2-4":0,"4-6":0,"6-8":0,"8-10":0, "10-15":0, "15-20":0, "20+":0},
			{"0-2":0,"2-4":0,"4-6":0,"6-8":0,"8-10":0, "10-15":0, "15-20":0, "20+":0}]


with open('train.pkl', 'rb') as f:
	train_data = pickle.load(f)

with open('valid.pkl', 'rb') as f:
	valid_data = pickle.load(f)

with open('test.pkl', 'rb') as f:
	test_data = pickle.load(f)


path_dir = "/home/shivang/code/Projects/Multimodal Emotion Recognition/segments/"
path = path_dir + '*.mp4'



train = 0
val = 0
test = 0


for file in glob.glob(path):

	# print (file)
	duration = get_len(file)
	file = str(file)
	file_name = file[len(path_dir): -6]
	if(file_name[len(file_name)-1] is '_'):
		file_name = file_name[:len(file_name)-1]
	print (file_name)
	print (duration)

	idx = -1
	if file_name in train_data:
		print("train")
		idx = 0
		train += 1

	if file_name in valid_data:
		print("val")
		idx = 1
		val += 1

	if file_name in test_data:
		print("test")
		idx = 2
		test += 1

	if duration < 10:
		for limit in range(0,11,2):
			if duration <= limit:
				# print(duration, limit/2-1)
				emo_bins[idx][bins[limit/2-1]] += 1
				break

	elif duration >= 10 and duration <= 20:
		for limit in range(10,21,5):
			if duration <= limit:
				# print(duration, limit/5-1)
				emo_bins[idx][bins[limit/5-1]] += 1
				break

	else:
		emo_bins[idx][bins[7]] += 1


	print("\n")



print("Train data = " + str(train))
print("Val data = " + str(val))
print("Test data = " +str(test))
print("\n")


for idx in range(3):
	print(splits[idx] + " length:")

	for limit in range(0,11,2):
		print(bins[limit/2-1]+" = "+ str(emo_bins[idx][bins[limit/2-1]]))

	for limit in range(10,21,5):
		print(bins[limit/5-1]+" = "+ str(emo_bins[idx][bins[limit/5-1]]))

	print(bins[7]+" = "+ str(emo_bins[idx][bins[7]]))



end_time = time.time()

print("Time taken = " + str(end_time-start_time))
