import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import librosa
# from tkinter import TclError


# constants
CHUNK = 320                 # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 9600                 # samples per second

# create matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(15, 7))

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,output=True,frames_per_buffer=CHUNK)

# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)

# create a line object with random data
line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)

# basic formatting for the axes
ax.set_title('AUDIO WAVEFORM')
ax.set_xlabel('samples')
ax.set_ylabel('volume')
ax.set_ylim(-1, 1)
ax.set_xlim(0, 2 * CHUNK)
plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-1, 0, 1])

# show the plot
plt.show(block=False)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()
total = 0
start = True
while True:
    
    # binary data
    if start:
        start_time = time.time()
        start =False

    data = stream.read(CHUNK)#, exception_on_overflow = False)  
    
    # audio_data = np.fromstring(data, dtype=np.float32)
    # print(audio_data.shape)
    # data_np = librosa.resample(audio_data, 9600*2, 9600)
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    
    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2]#/128 #+ 128
    # print(data_np)
    librosa.output.write_wav('1.wav', data_np, 9600)
    data_np,sr = librosa.load('1.wav',sr = 9600)
    # data_np = librosa.resample(data_np, 9600, 9600)
    total += 320
    print(data_np)
    # break
    # if (time.time()-start_time >= 1):
    #     print(total/CHUNK)
    #     break
    # print (data_np.shape)

    # ****** data_np IS WHAT GET'S FED INTO THE VOCAL PIPELINE FOR THE NEURAL NET ********
    
    line.set_ydata(data_np)
    
    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1
        
    except:
        break
        # # calculate average frame rate
        # frame_rate = frame_count / (time.time() - start_time)
        
        # print('stream stopped')
        # print('average frame rate = {:.0f} FPS'.format(frame_rate))
        # break
