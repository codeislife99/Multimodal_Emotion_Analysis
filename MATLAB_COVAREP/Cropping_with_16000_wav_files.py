

import glob
import librosa
import scipy.io.wavfile as read
import matplotlib.pyplot as plt
import os


# def crop(y,crop_length):
#   length = len(y)
#   start = length//2 - crop_length//2
#   return y[start:start+crop_length]
# # Replace WAV Files with new sampling rate
# crop_length = 24000
# test = [1,2]
# valid = [3,4]
# for actor_no in range(1,25,1):
#   actor_str = str(actor_no).zfill(2)
#   print(actor_str)
#   for filename in glob.glob("../Actor_"+actor_str+"/*.wav"):        
#       y,sr = librosa.load(filename,sr = 16000)
#       # if len(y)<24000:
#         # continue
#       # else:
#         # y = crop(y,crop_length)
#       # if(len(y)!=crop_length):
#         # print("Problem")
#       # else:
#       # print(filename[12:])
#       if actor_no in test:
#         librosa.output.write_wav('./WAVFiles/test/'+filename[12:], y, sr)
#       elif actor_no in valid:
#         librosa.output.write_wav('./WAVFiles/valid/'+filename[12:], y, sr)
#       else:
#         librosa.output.write_wav('./WAVFiles/train/'+filename[12:], y, sr)


# Verify new files
# for actor_no in range(1,25,1):
#   actor_str = str(actor_no).zfill(2)
for filename in glob.glob("./WAVFiles/test/*.wav"):        
    rate,_ = read.read(filename)
    print(filename,rate)
for filename in glob.glob("./WAVFiles/valid/*.wav"):        
    rate,_ = read.read(filename)
    print(filename,rate)
for filename in glob.glob("./WAVFiles/train/*.wav"):        
    rate,_ = read.read(filename)
    print(filename,rate)
