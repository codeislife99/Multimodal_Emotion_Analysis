import file_walker
import os
import cv2
import numpy as np
import imutils
import shutil
import random
import time

from multiprocessing.pool import Pool
from multiprocessing import cpu_count

src_dir = "./Processed Data/data/sample_large/train/"
dest_dir = "./Processed Data/data/sample_small/train/"

SEED = time.time()

def crawl(dir):
	print(dir)
	dst = dest_dir + dir + '/'
	dir = src_dir + dir

	video_dict = dict()

	cnt = 0;
	for f in file_walker.walk(dir):
		video_no = int(f.name[0:4])
		video_dict[video_no] = video_dict.get(video_no, []) + [f.full_path]
		
		# if ((frame_no % 6) == 0):
		# 	shutil.copy(f.full_path, dst)
		# 	cnt += 1
		

	for key, value in video_dict.items():
		random.seed(SEED)
		random.shuffle(value)
		print(key, ":\n")

		for val in value[0:8]:
			shutil.copy(val, dst)
			print(val)

		print("\n")

		# print(key, value)
		# print(key, len(value))
		
		cnt+= 1
		if cnt == 25:
			break



	# print(dir, "=>", cnt)

	# when running without multi-processing 

	"""
	for f in file_walker.walk(dir):
		if(f.isDirectory):
			emotion_dir = f.name
			dst = dest_dir + emotion_dir + '/'
			print(emotion_dir, ":")

			for sub_f in file_walker.walk(f.full_path):
				frame_no = int(sub_f.name[22:])

				if (frame_no >= 20 and frame_no <= 50):
					shutil.copy(sub_f.full_path, dst)
					cnt += 1
	"""


	return cnt


if __name__ == '__main__':

	pool = Pool(cpu_count()*2)
	dir_to_crawl = ['A/', 'H/', 'N/', 'D/', 'S/', 'F/']

	# dir_to_crawl = ['A/']
	count = 0
	try:
		for cnt in pool.imap_unordered(crawl, dir_to_crawl):
			count += cnt
			print (cnt)

	except Exception as e:
		print("exception:",e)
		pool.close()
		pool.terminate()

	print(count)

	

