import wave
import contextlib
import os
import glob
import numpy as np
import scipy.io as sio
bins = np.zeros((24,8), dtype = np.float32)
bins_dict= {0:"< 90", 1:"90-100", 2:"100-110", 3:"110-120",4:"120-130",5:"130-150",6:"150-180",7:"180+"}
for actor_no in range(1,25,1):
	actor_str = str(actor_no).zfill(2)
	# print(actor_str)
	for file in glob.glob("../Actor_"+actor_str+"/*.mat"):        
		# print(file)
		# path = file
		# # with contextlib.closing(wave.open(fname,'r')) as f:
		# #     frames = f.getnframes()
		# #     rate = f.getframerate()
		# #     duration = frames / float(rate)
		# #     print(duration)

		# f=open(path,"r")

		# #read the ByteRate field from file (see the Microsoft RIFF WAVE file format)
		# #https://ccrma.stanford.edu/courses/422/projects/WaveFormat/
		# #ByteRate is located at the first 28th byte
		# f.seek(28)
		# a=f.read(4)

		# #convert string a into integer/longint value
		# #a is little endian, so proper conversion is required
		# byteRate=0
		# for i in range(4):
		#     byteRate=byteRate + ord(a[i])*pow(256,i)

		# #get the file size in bytes
		# fileSize=os.path.getsize(path)  

		# #the duration of the data, in milliseconds, is given by
		# ms=((fileSize-44)*1000)/byteRate
		a = sio.loadmat(file)
		no_of_seconds = a['features'].shape[0]*0.01
		# no_of_frames = (ms*29.97)/1000
		no_of_frames = no_of_seconds*29.97
		# print(no_of_frames)
		if no_of_frames < 90:
			bins[actor_no-1][0] += 1
			print(no_of_frames)
		elif no_of_frames < 100:
			bins[actor_no-1][1] += 1
		elif no_of_frames < 110:
			bins[actor_no-1][2] += 1
		elif no_of_frames < 120:
			bins[actor_no-1][3] += 1
		elif no_of_frames < 130:
			bins[actor_no-1][4] += 1
		elif no_of_frames < 150:
			bins[actor_no-1][5] += 1
		elif no_of_frames < 180:
			bins[actor_no-1][6] += 1
		else:
			bins[actor_no-1][7] += 1
			print(no_of_frames)

		# print("File duration in miliseconds : " + str(ms))
		# print("File duration in H,M,S,mS : " +str(ms/(3600*1000))  "," str(ms/(60*1000)) "," str(ms/1000)  "," ms%1000)
		# print("Actual sound data (in bytes) : " % fileSize-44)  
		# f.close()
	print(bins.sum(0))
print(bins)
np.save('bins.npy',bins)
bins_sum = bins.sum(0)
print(bins_sum)
for i in range(0,8,1):
	print(bins_dict[i]+" = "+str(bins_sum[i]))


