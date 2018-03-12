

import glob
import librosa
import scipy.io.wavfile as read
import matplotlib.pyplot as plt
import os


def crop(y,crop_length):
  length = len(y)
  start = length//2 - crop_length//2
  return y[start:start+crop_length]
# Replace WAV Files with new sampling rate
crop_length = 24000
for filename in glob.iglob('../AudioWAV/*.wav'):
    y,sr = librosa.load(filename,sr = 16000)
    if len(y)<24000:
      continue
    else:
      y = crop(y,crop_length)
    if(len(y)!=crop_length):
      print("Problem")
    else:
      librosa.output.write_wav('./WAVFiles/'+filename[12:], y, sr)


# Verify new files
for i,filename in enumerate(glob.iglob('*/WAVFiles/*.wav')):
    rate,_ = read.read(filename)
    print(i,filename,rate)

