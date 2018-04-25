from __future__ import print_function, division, unicode_literals
import example_helper
import json
import csv
import numpy as np
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from multiprocessing import Process,Queue,Value,Array,Lock
import os
import time
import cv2
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
import h5py
from PIL import Image
import os
import random
import pickle
import time
import gc
import re
import time
import math
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import dlib
import vlc
import sys
import json
import webbrowser,os
import serial
import pyaudio
import wave
import matlab.engine
# import torch
# import torch.autograd as autograd
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import pickle
# import numpy as np
# import random
# import wordembed
# import padvec
# import multiprocessing
# import syslog
# import dlib
# import argparse
# import numpy as np
# import copy 
# import torchvision.models as models
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import os
# import shutil
# from PIL import Image
# import torch.nn.functional as F
# from threading import Thread
# import queue
# import sys
# from termios import tcflush, TCIOFLUSH
# import warnings
# warnings.filterwarnings("ignore")
# import serial
# import dlib
# from threading import Thread
# from queue import Queue


# def info(title):
#     print(title)
#     # print('module name:', __name__)
#     # print('parent process:', os.getppid())
#     # print('process id:', os.getpid())

# def f(name,q,shared_folder):
#     # info('function f')
#     # print('hello', name)
#     print(shared_folder.value)
#     q.put(name)
eng = matlab.engine.start_matlab("-desktop")

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

for port_no in range(10):

    port = '/dev/ttyACM'+str(port_no)
    try:
        string_ = "sudo chmod a+rw "+port+ " &>> garbage.txt"
        os.system(string_)
        ard = serial.Serial(port,57600,timeout=5)
        time.sleep(2) # wait for Arduino
        break
    except:
        pass

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def visual_vocal(flag):

    class LSTMAU(nn.Module):
        def __init__(self):
            super(LSTMAU, self).__init__()
            self.input_size = 17
            self.hidden_size = 64
            self.num_layers = 2
            self.no_of_emotions = 6
            self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers,bidirectional=True)
            self.linear = nn.Linear(2*self.hidden_size, self.no_of_emotions)
            self.bn = nn.BatchNorm1d(45)


        def forward(self,x):
            # print(x.size())
            x= self.bn(x)
            # x = x.unsqueeze(1)
            x = torch.transpose(x,0,1)
            # print(x.size())
            hiddens,_ = self.lstm(x)
            outputs = self.linear(hiddens[-1])
            # print(outputs)

            return outputs

    class VisionNet(nn.Module):
        def __init__(self,input_size,hidden_size,num_layers,no_of_emotions):
            super(VisionNet, self).__init__()
            self.lstm = nn.LSTM(17,128,num_layers,bidirectional=False)


        def forward(self,x):
            x = torch.transpose(x,0,1)
            hiddens,_ = self.lstm(x)
            hiddens = hiddens.squeeze(1)
            return hiddens
    '---------------------------------------------------LSTM VocalNet-------------------------------------------------------'

    class VocalNet(nn.Module):
        def __init__(self,input_size,hidden_size,num_layers,no_of_emotions):
            super(VocalNet, self).__init__()
            self.lstm = nn.LSTM(74,128,num_layers,bidirectional=False)
            self.linear = nn.Linear(hidden_size, no_of_emotions)


        def forward(self,x):
            x = torch.transpose(x,0,1)
            hiddens,_ = self.lstm(x)
            hiddens = hiddens.squeeze(1)
            return hiddens
    '---------------------------------------------------Dual Attention----------------------------------------------------'

    class DualAttention(nn.Module):
        def __init__(self,no_of_emotions):
            super(DualAttention, self).__init__()
            N = 128 
            ''' K= 1 ''' 
            self.Wvision_1 = nn.Linear(N,N)
            self.Wvision_m1 = nn.Linear(N,N)
            self.Wvision_h1 = nn.Linear(N,N)
            self.Wvocal_1 = nn.Linear(N,N)
            self.Wvocal_m1 = nn.Linear(N,N)
            self.Wvocal_h1 = nn.Linear(N,N)

            ''' K = 2 '''
            self.Wvision_2 = nn.Linear(N,N)
            self.Wvision_m2 = nn.Linear(N,N)
            self.Wvision_h2 = nn.Linear(N,N)
            self.Wvocal_2 = nn.Linear(N,N)
            self.Wvocal_m2 = nn.Linear(N,N)
            self.Wvocal_h2 = nn.Linear(N,N)

            self.fc = nn.Linear(N, no_of_emotions)


        def forward(self,vocal,vision):
            # Sorting out vision
            # print(resnet_output.size())
            # resnet_output = resnet_output.mean(0)
            # resnet_output = resnet_output.view(512,49)
            # vision = resnet_output.transpose(0,1)

            '-------------------------------------------------Initializing Memory--------------------------------------'

            vision_zero = vision.mean(0).unsqueeze(0)
            vocal_zero = vocal.mean(0).unsqueeze(0)
            m_zero = vision_zero * vocal_zero 
            m_zero_vision = m_zero.repeat(vision.size(0),1)
            m_zero_vocal = m_zero.repeat(vocal.size(0),1)
            '--------------------------------------------------K = 1 ---------------------------------------------------'
            # Visual Attention
            h_one_vision = F.tanh(self.Wvision_1(vision))*F.tanh(self.Wvision_m1(m_zero_vision))
            a_one_vision = F.softmax(self.Wvision_h1(h_one_vision),dim = -1)
            vision_one = (a_one_vision*vision).mean(0).unsqueeze(0)

            # Vocal Attention
            h_one_vocal = F.tanh(self.Wvocal_1(vocal))*F.tanh(self.Wvocal_m1(m_zero_vocal))
            a_one_vocal = F.softmax(self.Wvocal_h1(h_one_vocal),dim = -1)
            vocal_one = (a_one_vocal*vocal).mean(0).unsqueeze(0)

            # Memory Update
            m_one = m_zero + vision_one * vocal_one 
            m_one_vision = m_one.repeat(vision.size(0),1)
            m_one_vocal = m_one.repeat(vocal.size(0),1)

            '--------------------------------------------------K = 2  ---------------------------------------------------'

            # Visual Attention
            h_two_vision = F.tanh(self.Wvision_2(vision))*F.tanh(self.Wvision_m2(m_one_vision))
            a_two_vision = F.softmax(self.Wvision_h2(h_two_vision),dim = -1)

            vision_two = (a_two_vision*vision).mean(0).unsqueeze(0)
            # Vocal Attention
            h_two_vocal = F.tanh(self.Wvocal_2(vocal))*F.tanh(self.Wvocal_m2(m_one_vocal))
            a_two_vocal = F.softmax(self.Wvocal_h2(h_two_vocal),dim = -1)
            vocal_two = (a_two_vocal*vocal).mean(0).unsqueeze(0)

            # Memory Update
            m_two = m_one + vision_two * vocal_two 
            '-------------------------------------------------Prediction--------------------------------------------------'

            outputs = self.fc(m_two)
            # print(outputs)
            return outputs  
    '------------------------------------------------------Set Up -----------------------------------------------------------------'
    decoder = LSTMAU()
    decoder = decoder.cuda()
    decoder.train(False)
    criterion = nn.CrossEntropyLoss()
    # pretrained_file = '/media/teamd/New Volume/CREMA-D/AudioWAV_1600/AUOFLSTMNet/AULSTM_net_iter__10.pth.tar'
    # checkpoint = torch.load(pretrained_file)
    # decoder.load_state_dict(checkpoint['decoder'])
    emo_dict = {0:"Anger", 1:"Disgust", 2:"Fear", 3:"Happiness", 4:"Neutral", 5:"Sad"}
    ag = list(range(45))
    cols = list(range(5,22,1))
    no_of_emotions = 6
    vocal_input_size = 74 # Dont Change
    vision_input_size = 17 # Dont Change
    vocal_num_layers = 2
    vision_num_layers = 2
    vocal_hidden_size = 128
    vision_hidden_size = 128
    Vocal_encoder = VocalNet(vocal_input_size, vocal_hidden_size,vocal_num_layers,no_of_emotions)
    Vision_encoder = VisionNet(vision_input_size, vision_hidden_size,vision_num_layers,no_of_emotions)
    Attention = DualAttention(no_of_emotions)
    pretrained_file = 'DAN.pth.tar'
    # pretrained_file = 'attention_net__0.pth.tar'

    checkpoint = torch.load(pretrained_file)
    Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
    Vision_encoder.load_state_dict(checkpoint['Vision_encoder'])
    Attention.load_state_dict(checkpoint['Attention'])
    Vocal_encoder.eval()
    Vision_encoder.eval()
    Attention.eval()
    while True:
        if flag.value == 1:
            break
    for idx in range(24):
        # start_time = time.time()
        with open('COVAREP_feature_extraction.m','r') as f:
            data = f.readlines()
        data[50] = "in_dir = "+"'/media/teamd/New Volume/CREMA-D/AudioWAV_1600/audio_recordings/output"+str(idx)+"/'"+"\n" # Put your path here
        with open("COVAREP_feature_extraction.m",'w') as file:
            file.writelines(data)
        eng.COVAREP_feature_extraction(nargout = 0)
        vocal_seq_input = np.empty((1,vocal_seq_len,vocal_input_size), dtype = np.float32)
 
        vocal_mat_file = "./AudioWAV_1600/audio_recordings/output"+str(i)+"/"+str(i)+".mat" # Put your path here
        struct = sio.loadmat(vocal_mat_file)
        struct['features'][[struct['features']<(-1e8)]] = 0
        vocal_seq_input[0] = struct['features'][0:150]
        vocal_seq_i = Variable(torch.from_numpy(vocal_seq_input)).cuda()
        vocal_output = Vocal_encoder(vocal_seq_i)

        os.system("/media/teamd/New\ Volume/OF/build/bin/FeatureExtraction -aus -fdir ./AudioWAV_1600/Trainings/Train_batch_"+str(idx)) # Put your path here 
        csv_file_path = './processed/Train_batch_'+str(idx)+'.csv'
        df = pd.read_csv(csv_file_path)
        df = df.iloc[ag,cols]
        target_df = np.array(df.values , dtype = np.float32)
        resnet_output = Variable(torch.from_numpy(target_df), volatile = True).cuda()
        resnet_output = resnet_output.unsqueeze(0)  
        # outputs = decoder(resnet_output)
        vision_output = Vision_encoder(resnet_output)
        outputs = Attention(vocal_output,vision_output)

        visual_vocal_probs = F.softmax(outputs, dim = 1).data.cpu().numpy()[0]

        for visual_vocal_index in range(6):
            visual_vocal_arr[current_visual_seq.value*6 + visual_vocal_index] = visual_vocal_probs[visual_vocal_index]

        current_visual_seq.value += 1
        _, preds = torch.max(outputs.data, 1)
        emotion_no = preds.cpu().numpy()[0]
        emotion = emo_dict[emotion_no]
        # print(emotion)
        # print("Time Taken = " ,time.time() -start_time)    

def webcam(shared_folder,flag,q,mutex):
    vc = cv2.VideoCapture(-1)
    no_of_frames = 0
    detector = dlib.get_frontal_face_detector()
    start_time = time.time()
    total_frames = 0
    delayVal = 0
    while vc.isOpened():
        _, img_frame = vc.read()
        img = cv2.flip(img_frame, 1)
        # img = adjust_gamma(img, 1.5)
        # try:
        #     dets = detector(img)
        #     # for i, d in enumerate(dets):
        #     d = dets[0]                   
        #     cv2.rectangle(img, (d.left(), d.top()), ( d.right(), d.bottom()), (255,0,0), 2)
        #     centroid=((d.left()+d.right())/2, (d.top()+ d.bottom())/2)
        #     val1x=centroid[0]
        #     val1y=centroid[1]
        #     with mutex:
        #         if (delayVal % 3 == 0):
        #             q.put(val1x)                    
        #             q.put(val1y)


        #     delayVal += 1

        # except:
        #     pass

        if no_of_frames%2 == 0:
            # print(no_of_frames%90)
            cv2.imwrite('./AudioWAV_1600/Trainings/Train_batch_'+str(shared_folder.value)+'/'+str(int((no_of_frames%90)/2)).zfill(2)+'.jpg', img_frame)
        cv2.putText(img_frame, str(TEST_SENTENCES[shared_folder.value]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

        cv2.imshow("Original Frame", img_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vc.release()
            cv2.destroyAllWindows()
            break

        no_of_frames = (no_of_frames+1)%90
        total_frames += 1
        if no_of_frames== 0:
            shared_folder.value = (shared_folder.value+1)
            sm_bool[shared_folder.value-1] = 0
            os.system("mkdir -p ./AudioWAV_1600/Trainings/Train_batch_"+str(shared_folder.value))
            # path = "./videos/video"+str(shared_folder.value)+".mp4"
            # os.system("ffmpeg -framerate 15 -i ./AudioWAV_1600/Trainings/Train_batch_"+str(shared_folder.value-1)+"/%02d.jpg -loglevel quiet -c:v libx264  -crf 20 -pix_fmt yuv420p "+str(path))
            flag.value = 1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return (np.exp(x) / np.exp(x).sum())

def predict():

    # Text_Emotion_Dict ={0: 'love/relief', 1:'enthusiasm/surprise/happiness/joy/fun', 2:'hate/anger/fear', 3:'empty/sadness/worry/boredom', 4:'neutral'}
    # Vision/Vocal_Emotion_Dict = {0:"Anger", 1:"Disgust", 2:"Fear", 3:"Happiness", 4:"Neutral", 5:"Sad"}

    Emotion={0: 'Content', 1:'Happiness', 2:'Anger/Hate', 3:'Sad', 4:'Neutral'}
    
    vv_arr = list(range(24))

    C= 0 
    vv_correct = 0.0
    text_correct = 0.0
    while True:
        # print(current_text_seq.value)
        probs = np.array([0.0, 0.0 ,0.0,0.0,0.0]) 
        vv_probs = np.array([0.0, 0.0, 0.0,0.0 ,0.0])
        text_probs = np.array([0.0,0.0,0.0,0.0,0.0])

        if current_visual_seq.value>C and current_text_seq.value>C: 
            # for _i in range(5):
            #     print(text_arr[C*5+_i])
            alpha = 0.5
            for i_ in range(5):
                text_probs[i_] = text_arr[C*5+i_]
            text_pred = np.argmax(text_probs)

            vv_probs[0] = text_arr[C*5+0] # Content
            vv_probs[1] = visual_vocal_arr[C*6+3] # Happiness 
            vv_probs[2] = visual_vocal_arr[max(C*6+0,max(C*6+1,C*6+2))] # Anger/Disgust/Fear
            vv_probs[3] = visual_vocal_arr[C*6+5]  # Sad
            vv_probs[4] = visual_vocal_arr[C*6+4]  # Neutral
            vv_probs = softmax(vv_probs) 
            # print(vv_probs)
            vv_pred = np.argmax(vv_probs)

            probs[0] = text_arr[C*5+0] # Content
            probs[1] = alpha*visual_vocal_arr[C*6+3]+(1-alpha)*text_arr[C*5+1] # Happiness 
            probs[2] = alpha*visual_vocal_arr[max(C*6+0,max(C*6+1,C*6+2))] + (1-alpha)*text_arr[C*5+2] # Anger/Disgust/Fear
            probs[3] = alpha*visual_vocal_arr[C*6+5] + (1-alpha)*text_arr[C*5+3] # Sad
            probs[4] = alpha*visual_vocal_arr[C*6+4] + (1-alpha)*text_arr[C*5+4] #+ 0.2 # Neutral
            pred_arr[C] = np.argmax(probs)

            if pred_arr[C] == groundtruth[C]:
                correct.value += 1
            if vv_pred == groundtruth[C]:
                vv_correct += 1
            if text_pred == groundtruth[C]:
                text_correct += 1
            emotion_string = Emotion[pred_arr[C]]
            vv_string = Emotion[vv_pred]
            text_string = Emotion[text_pred]
            # sum_probs = softmax(probs)
            UI_dict = {}
            UI_dict['prediction'] = probs.tolist()
            UI_dict['max'] = emotion_string
            UI_dict['transcript'] = str(TEST_SENTENCES[C])
            UI_dict['acc'] = str(correct.value/(1.0*(C+1)))
            UI_dict['vv_max'] = str(vv_string)
            UI_dict['text_max'] = str(text_string)
            UI_dict['vv_acc'] = str(vv_correct/(C+1))
            UI_dict['text_acc'] = str(text_correct/(C+1))
            UI_dict['vv'] = vv_probs.tolist()
            UI_dict['text'] = text_probs.tolist()
            json_file_name = './predictions/json'+str(C+1)+'.json'
            with open(json_file_name, 'w') as fp:
                json.dump(UI_dict, fp)
            # sm_bool[C] = 0
            text_file=open('Script.txt',mode ='r', encoding='utf-8-sig')
            # print(probs)
            for idx,line in enumerate(text_file):
                if idx == C:
                    print(line[:-1],Emotion[np.argmax(probs)])
                    break
            C+=1
            current_ov_seq.value += 1

            if C == 21:
                webbrowser.open("file://"+"/media/teamd/New Volume/CREMA-D/index.html")

def save_video():
    prev_number = 0
    while True:
        number = current_ov_seq.value+1
        if number != prev_number:
            path = "./videos/video"+str(number)+".mp4"
            os.system("ffmpeg -framerate 15 -i ./AudioWAV_1600/Trainings/Train_batch_"+str(number-1)+"/%02d.jpg -loglevel quiet -c:v libx264  -crf 20 -pix_fmt yuv420p "+str(path))
            prev_number = number            

def record():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 3
    blockPrint()
    for i in range(24):
        os.system("mkdir -p ./AudioWAV_1600/audio_recordings/output"+str(i)+"/")

        WAVE_OUTPUT_FILENAME = "./AudioWAV_1600/audio_recordings/output"+str(i)+"/"+str(i)+".wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            # print(data)
            frames.append(data)

        print("* done recording")

        start_time = time.time()
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        # time.sleep(1.28)
    enablePrint()

# Disable

def statemachine():
    Emotion={0: 'Content', 1:'Happiness', 2:'Anger/Hate', 3:'Sad', 4:'Neutral'}
    prev = -1
    prev_count = 0
    limit_ = 1
    while True:
        C = shared_folder.value - 1
        # print(C)
        if C>=0 and sm_bool[C] == 0:
            sm_bool[C] = 1
            current = text_pred_arr[C]

            if prev==current:
                prev_count+=1
                if (prev_count > 0) and ((prev_count % limit_) is 0):
                    if current == 1:
                        num = np.random.choice([1,2])
                        filename = "./AudioWAV_1600/StateMachine recordings/happy_"+str(num)+".mp3"
                    elif current == 0:
                        filename = "./AudioWAV_1600/StateMachine recordings/content_1.mp3"
                    elif current == 2:
                        filename = "./AudioWAV_1600/StateMachine recordings/anger_1.mp3"
                    elif current == 3:  
                        num = np.random.choice([1,2])
                        filename = "./AudioWAV_1600/StateMachine recordings/sad_"+str(num)+".mp3"
                    else:
                        filename = False
                    if filename:
                        p = vlc.MediaPlayer(filename)
                        blockPrint()
                        p.play()
                        enablePrint()
            else:
                prev_count = 0
                prev = current



# def sendtoard(q, mutex):
#     condition = 1
#     while (1):
#     # Serial write section

#         if (condition == 2):
#             try:
#                 #ard.flush()
#                 condition = 1
#                 # print (condition)
#             except:
#                 pass
#         #print ('-------------------------')
#         #print("acquiring_mutex_serial")
#         with mutex:
#             #print("acquired_mutex_serial")
#             if (q.qsize() > 4):
#                 while (q.qsize() > 2):
#                     garbage = q.get()

#             # print (q.qsize())
#             if q.qsize() > 2:
#                 val2x = q.get()
#                 val2y = q.get()
#                 # print (format(int(val2x), '03d')+format(int(val2y), '03d'))
        
#                 if (condition == 1):
#                     try:
#                         # print ((str(val2x)+","+str(val2y)+"_").encode('utf-8'))
#                         ard.write((str(int(val2x))+","+str(int(val2y))+"_").encode('utf-8'))
#                         condition = 2
#                     except:
#                         pass
#             # print (q.qsize())
#         #print("released_mutex_serial")

        
#         time.sleep(0.15) # I shortened this to match the new value in your Arduino code

        
#     else:
#         pass
#         # print ("Exiting")
#     exit()

def accuracy():
    time.sleep(250)
    print("Overall Accuracy = ", str(correct.value/total))

def get_emoprob(emoji_prob):
    emo_prob = [0.0, 0.0 ,0.0 ,0.0 ,0.0]
    for i in range(len(emoji_prob)):
        if i in HAPPY:
            emo_prob[1]+= emoji_prob[i]
        elif i in SAD:
            emo_prob[3]+= emoji_prob[i]
        elif i in ANGER:
            emo_prob[2]+= emoji_prob[i]
        elif i in LOVE:
            emo_prob[0]+= emoji_prob[i]
        else:
            emo_prob[4]+= emoji_prob[i]
    return emo_prob

TEST_SENTENCES = []
#Emotion={0: 'Content', 1:'Happiness', 2:'Anger/Hate', 3:'Sad', 4:'Neutral'}
groundtruth = [1,1,1,1,1,1,0,0,0,0,3,3,3,3,3,4,4,4,4,4,2,2,2,2]

if __name__ == '__main__':
    # info('main line')
    try:
        os.system("rm ./predictions/*.json")
        os.system("rm ./videos/*.mp4")
    except:
        pass

    f= open('Script.txt',mode ='r', encoding='utf-8-sig')
    for idx,line in enumerate(f):
        # print(line)
        TEST_SENTENCES.append(line[:-1])
    # print(TEST_SENTENCES)
    maxlen = 30

    p = vlc.MediaPlayer('./AudioWAV_1600/StateMachine recordings/welcome.mp3')
    blockPrint()
    p.play()
    enablePrint()
    time.sleep(12)

    q = Queue()
    mutex = Lock()

    shared_folder = Value('i', 0)
    flag = Value('i', 0)
    pred_arr = Array('i', range(24))
    sm_bool = Array('i', range(24))
    visual_vocal_arr = Array('d', range(144))
    correct = Value('i', 0)
    text_arr = Array('d', range(120))
    text_pred_arr = Array('i',range(24))
    current_visual_seq = Value('i', 0)
    current_text_seq = Value('i', 0)
    correct = Value('i', 0)
    current_ov_seq = Value('i',-1)
    total = 24  

    # p3 = Process(target=sendtoard, args=((q),mutex))
    p4 = Process(target=webcam, args=(shared_folder,flag,q,mutex,))
    p5 = Process(target=visual_vocal, args=(flag, ))
    p6 = Process(target=predict, args =())
    p7 = Process(target=statemachine, args=())
    p8 = Process(target=save_video, args=())
    p9 = Process(target=accuracy, args=())

    # p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()

    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, maxlen)

    model = torchmoji_emojis(PRETRAINED_PATH)

    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    prob = model(tokenized)

    HAPPY = [0,7,10,11,13,15,16,17,36,53,54,62,63] # 13 elements
    SAD = [2,3,5,27,29,34,43,46] # 8 elements 
    ANGER = [12,28,32,37,39,44,52,55,56,58] # 10 elements
    LOVE = [4,8,18,20,23,24,31,35,40,47,50,59,60,61] # 14 elements
    NEUTRAL = [1,6,9,14,19,21,22,25,26,30,33,38,41,42,45,48,49,51,57] # 19 elements
    Emo_dict = {0: 'Content', 1:'Happiness', 2:'Anger/Hate', 3:'Sad', 4:'Neutral'}


    for prob in [prob]:
        scores = []
        for i, t in enumerate(TEST_SENTENCES):

            t_tokens = tokenized[i]
            t_score = [t]
            t_prob = prob[i]
            emo_prob_ = np.around(np.array(get_emoprob(t_prob)), decimals=4)
            t_score.append(Emo_dict[np.argmax(emo_prob_)])
            t_score.extend(emo_prob_)

            for text_idx in range(5):
                text_arr[current_text_seq.value*5 + text_idx] = emo_prob_[text_idx] 
            text_pred_arr[current_text_seq.value] = np.argmax(emo_prob_)
            current_text_seq.value += 1


