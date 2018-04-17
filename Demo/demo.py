from __future__ import print_function, division, unicode_literals
import example_helper
import json
import csv
import numpy as np
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from multiprocessing import Process,Queue,Value,Array
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

q = Queue()

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
    '------------------------------------------------------Set Up -----------------------------------------------------------------'
    decoder = LSTMAU()
    decoder = decoder.cuda()
    decoder.train(False)
    criterion = nn.CrossEntropyLoss()
    pretrained_file = '/media/teamd/New Volume/CREMA-D/AudioWAV_1600/AUOFLSTMNet/AULSTM_net_iter__10.pth.tar'
    checkpoint = torch.load(pretrained_file)
    decoder.load_state_dict(checkpoint['decoder'])
    emo_dict = {0:"Anger", 1:"Disgust", 2:"Fear", 3:"Happiness", 4:"Neutral", 5:"Sad"}
    ag = []
    for jk in range(45):
        ag.append(jk)

    while True:
        if flag.value == 1:
            break
    for idx in range(30):
        # start_time = time.time()
        os.system("/media/teamd/New\ Volume/OF/build/bin/FeatureExtraction -aus -fdir ./AudioWAV_1600/Trainings/Train_batch_"+str(idx))
        csv_file_path = './processed/Train_batch_'+str(idx)+'.csv'
        df = pd.read_csv(csv_file_path)
        df = df.iloc[ag,[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
        target_df = np.array(df.values , dtype = np.float32)
        resnet_output = Variable(torch.from_numpy(target_df), volatile = True).cuda()
        resnet_output = resnet_output.unsqueeze(0)  
        outputs = decoder(resnet_output)
        visual_vocal_probs = F.softmax(outputs, dim = 1).data.cpu().numpy()[0]

        for visual_vocal_index in range(6):
            visual_vocal_arr[current_visual_seq.value*6 + visual_vocal_index] = visual_vocal_probs[visual_vocal_index]

        current_visual_seq.value += 1
        _, preds = torch.max(outputs.data, 1)
        emotion_no = preds.cpu().numpy()[0]
        emotion = emo_dict[emotion_no]
        # print(emotion)
        # print("Time Taken = " ,time.time() -start_time)    

def webcam(shared_folder,flag):
    vc = cv2.VideoCapture(-1)
    no_of_frames = 0
    detector = dlib.get_frontal_face_detector()
    start_time = time.time()
    total_frames = 0
    while vc.isOpened():
        _, img_frame = vc.read()
        if no_of_frames%2 == 0:
            # print(no_of_frames%90)
            cv2.imwrite('./AudioWAV_1600/Trainings/Train_batch_'+str(shared_folder.value)+'/'+str(int((no_of_frames%90)/2)).zfill(2)+'.jpg', img_frame)
        cv2.putText(img_frame, str(TEST_SENTENCES[shared_folder.value]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

        # try:
        #     dets = detector(img_frame)
        #     #print("Number of faces detected: {}".format(len(dets)))

        #     for i, d in enumerate(dets):                    
        #         #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #            # i, d.left(), d.top(), d.right(), d.bottom()))
        #         cv2.rectangle(img_frame, (d.left(), d.top()), ( d.right(), d.bottom()), (255,0,0), 2)
        #         centroid=((d.left()+d.right())/2, (d.top()+ d.bottom())/2)
        #         val1x=centroid[0]
        #         val1y=centroid[1]
        #         # with q.mutex:
        #         #     q.queue.clear()
        #         q.put(val1x)                    
        #         q.put(val1y)
        #         #print(val1)
                
        #         #the frame seems to have 620 horizontal pixels
        #         #the val tells you how many pixels from the centre he centroid of face is ocated.
        #         #-ve means image centroid is at the right f face centroid
        #         #+ve means viceversa
        #         #magnitude of val indicates how shifted it is
        # except:
        #     raise
        cv2.imshow("Original Frame", img_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vc.release()
            cv2.destroyAllWindows()
            break

        no_of_frames = (no_of_frames+1)%90
        total_frames += 1
        if no_of_frames== 0:
            shared_folder.value = (shared_folder.value+1)
            os.system("mkdir -p ./AudioWAV_1600/Trainings/Train_batch_"+str(shared_folder.value))
            # path = "./videos/video"+str(shared_folder.value)+".mp4"
            # os.system("ffmpeg -framerate 15 -i ./AudioWAV_1600/Trainings/Train_batch_"+str(shared_folder.value-1)+"/%02d.jpg -loglevel quiet -c:v libx264  -crf 20 -pix_fmt yuv420p "+str(path))
            flag.value = 1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict():

    # Text_Emotion_Dict ={0: 'love/relief', 1:'enthusiasm/surprise/happiness/joy/fun', 2:'hate/anger/fear', 3:'empty/sadness/worry/boredom', 4:'neutral'}
    # Vision/Vocal_Emotion_Dict = {0:"Anger", 1:"Disgust", 2:"Fear", 3:"Happiness", 4:"Neutral", 5:"Sad"}

    Emotion={0: 'Content', 1:'Happiness', 2:'Anger/Hate', 3:'Sad', 4:'Neutral'}
    C= 0 
    while True:
        # print(current_text_seq.value)
        probs = np.array([0.0, 0.0 ,0.0,0.0,0.0]) 
        if current_visual_seq.value>C and current_text_seq.value>C: 
            # for _i in range(5):
            #     print(text_arr[C*5+_i])
            alpha = 0
            probs[0] = text_arr[C*5+0] # Content
            probs[1] = alpha*visual_vocal_arr[C*6+3]+(1-alpha)*text_arr[C*5+1] # Happiness 
            probs[2] = alpha*visual_vocal_arr[max(C*6+0,max(C*6+1,C*6+2))] + (1-alpha)*text_arr[C*5+2] # Anger/Disgust/Fear
            probs[3] = alpha*visual_vocal_arr[C*6+5] + (1-alpha)*text_arr[C*5+3] # Sad
            probs[4] = alpha*visual_vocal_arr[C*6+4] + (1-alpha)*text_arr[C*5+4] #+ 0.2 # Neutral
            pred_arr[C] = np.argmax(probs)
            if pred_arr[C] == groundtruth[C]:
                correct.value += 1

            emotion_string = Emotion[pred_arr[C]]
            # sum_probs = softmax(probs)
            UI_dict = {}
            UI_dict['prediction'] = probs.tolist()
            UI_dict['max'] = emotion_string
            UI_dict['transcript'] = str(TEST_SENTENCES[C])
            json_file_name = './predictions/json'+str(C+1)+'.json'
            with open(json_file_name, 'w') as fp:
                json.dump(UI_dict, fp)
            sm_bool[C] = 0
            text_file=open('Script.txt',mode ='r', encoding='utf-8-sig')
            # print(probs)
            for idx,line in enumerate(text_file):
                if idx == C:
                    print(line[:-1],Emotion[np.argmax(probs)])
                    break
            C+=1
            current_ov_seq.value += 1

            if C == 15:
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
    pass

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def statemachine():
    Emotion={0: 'Content', 1:'Happiness', 2:'Anger/Hate', 3:'Sad', 4:'Neutral'}
    prev = -1
    prev_count = 0
    limit_ = 2
    while True:
        C = current_ov_seq.value
        if C>=0 and sm_bool[C] == 0:
            sm_bool[C] = 1
            current = pred_arr[C]
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


def sendtoard():
    try_ports = ['/dev/ttyACM0','/dev/ttyACM1','/dev/ttyACM2','/dev/ttyACM3']
    success = 0
    for port in try_ports:
        portx = '/dev/ttyACM2'
        os.system("sudo chmod a+rw "+portx)
        try:
            ard = serial.Serial(portx,9600,timeout=5)
            success = 1
            break
        except:
            raise
    if success == 0:
        print("COULDNT DO ARDUINO")
    time.sleep(2) # wait for Arduino
    condition = 1
    while (1):
    # Serial write section
        print (str(condition) + " condition")
        if (condition == 2):
            try:
                ard.flush()
                condition = 1
                print (condition)
            except:
                pass
        print("Sending1")
        print (condition)
        val2x = q.get()
        val2y = q.get()
        print ("Python value sent: ")
        print (format(int(val2x), '03d')+format(int(val2y), '03d'))
        

        print ("cake")
        print (val2x)
        print (val2y)
        print ("!!")
        print (condition)
        if (condition == 1):
            try:
                print ("#@@#")
                print ((str(val2x)+","+str(val2y)+"_").encode('utf-8'))
                ard.write((str(val2x)+","+str(val2y)+"_").encode('utf-8'))
                condition = 1
            except:
                pass
        time.sleep(0.20) # I shortened this to match the new value in your Arduino code

        # Serial read section
        #msg = ard.read(ard.inWaiting()) # read all characters in buffer
        #print ("Message from arduino: ")
        #msg=msg.decode("utf-8") 
        #print (msg)
        
    else:
        print ("Exiting")
    exit()

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
groundtruth = [1,1,1,1,1,1,0,0,0,0,0,0,3,3,3,3,3,3,3,4,4,4,4,4,4,2,2,2,2,2]
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

    # p = vlc.MediaPlayer('./AudioWAV_1600/StateMachine recordings/welcome.mp3')
    # blockPrint()
    # p.play()
    # enablePrint()
    # time.sleep(12)


    shared_folder = Value('i', 0)
    flag = Value('i', 0)
    pred_arr = Array('i', range(30))
    sm_bool = Array('i', range(30))
    visual_vocal_arr = Array('d', range(180))
    correct = Value('i', 0)
    text_arr = Array('d', range(150))
    current_visual_seq = Value('i', 0)
    current_text_seq = Value('i', 0)
    correct = Value('i', 0)
    current_ov_seq = Value('i',-1)
    total = 30.0  

    p4 = Process(target=webcam, args=(shared_folder,flag, ))
    p5 = Process(target=visual_vocal, args=(flag, ))
    p6 = Process(target=predict, args =())
    p7 = Process(target=statemachine, args=())
    p8 = Process(target=save_video, args=())
    p9 = Process(target=accuracy, args=())

    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()

    # print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, maxlen)

    # print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_emojis(PRETRAINED_PATH)
    # print(model)
    # print('Running predictions.')
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    prob = model(tokenized)

    HAPPY = [0,7,10,11,13,15,16,17,36,53,54,62,63] # 13 elements
    SAD = [2,3,5,27,29,34,43,46] # 8 elements 
    ANGER = [12,28,32,37,39,44,52,55,56,58] # 10 elements
    LOVE = [4,8,18,20,23,24,31,35,40,47,50,59,60,61] # 14 elements
    NEUTRAL = [1,6,9,14,19,21,22,25,26,30,33,38,41,42,45,48,49,51,57] # 19 elements
    # Emo_dict = {0:"Happy", 1:"Sad", 2:"Anger", 3:"Love", 4:"Neutral"}
    Emo_dict = {0: 'Content', 1:'Happiness', 2:'Anger/Hate', 3:'Sad', 4:'Neutral'}


    for prob in [prob]:
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the torchMoji repo.
        # print('Writing results to {}'.format(OUTPUT_PATH))
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
            current_text_seq.value += 1
    # p4.join()
    # p5.join()
    # p6.join()
    # p7.join()
    # p8.join()

    # if current_visual_seq == 29:
    #     print("Overall Accuracy = ", str(correct.value/total))
            # t_score.append(sum(t_prob[ind_top]))
            # t_score.extend(ind_top)
            # t_score.extend([t_prob[ind] for ind in ind_top])
            # scores.append(t_score)
            # print(t_score)

    # p6.start()
    # for i in range(1000):
        # q1 = Queue()
        # q2 = Queue()
        # q3 = Queue()
        # p1 = Process(target=f, args=('bob',q1,shared_folder))
        # p2 = Process(target=f, args=('bob2',q2,shared_folder))
        # p3 = Process(target=f, args=('bob3',q3,shared_folder))

        # p1.start()
        # p2.start()
        # p3.start()
        # print(q1.get())
        # print(q2.get())
        # print(q3.get())
        # p1.join()
        # p2.join()
        # p3.join()
        # print(i)

    # class LSTMEncoder(nn.Module):
    #     def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
    #         super(LSTMEncoder, self).__init__()
    #         self.input_dim = input_dim
    #         self.hidden_dim = hidden_dim  
    #         self.num_layers=num_layers
    #         self.num_classes=num_classes
    #         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)               
    #         self.linear1 = nn.Linear(in_features=hidden_dim * seq_len, out_features=64)
    #         self.linear2 = nn.Linear(in_features=64, out_features=32)
    #         self.linear3 = nn.Linear(in_features=32, out_features=num_classes)         
    #         self.hidden = self.init_hidden()

    #     def init_hidden(self):        
    #         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    #         return (autograd.Variable(torch.randn(self.num_layers, 1, self.hidden_dim)).cuda(),#2 for bidirectional
    #                 autograd.Variable(torch.randn(self.num_layers, 1, self.hidden_dim)).cuda())
            

    #     def forward(self, sentence):        
    #         lstm_out, self.hidden = self.lstm(sentence.view(seq_len, 1, self.input_dim), self.hidden)
    #         output = self.linear1(lstm_out.view(-1, self.hidden_dim *seq_len))
    #         output = F.leaky_relu(output)
    #         output = self.linear2(output)       
    #         output = F.leaky_relu(output)
    #         output1 = self.linear3(output)
    #         # output2 = F.log_softmax(output1, dim=1)     
    #         return output1

    # def hook(self, input, output):  
    #     i=1

    # seq_len=15
    # input_dim = 300
    # hidden_dim = 128
    # num_layers = 1
    # learning_rate=0.0001
    # no_epochs=3
    # num_classes=5


    # model=LSTMEncoder(input_dim, hidden_dim, num_layers, num_classes).cuda()
    # diction=torch.load('modelagain_epoch4_23000')
    # loss_function = nn.NLLLoss().cuda()
    # model.load_state_dict(diction['state_dict'])

    # '''
    # d = list(zip(test_samples, test_targets))
    # random.shuffle(d)
    # test_samples, test_targets = zip(*d)'''

    # #confusion=np.zeros((5,5), dtype=int)
    # padsize=15
    # # while True:
    # text_file=open("Script.txt")
    # for line in text_file:
    #     # print(line)
    #     try:
    #         tokens, vectors=wordembed.sentenceembed(line)
    #         padded=autograd.Variable(torch.Tensor(padvec.padded_vectors(vectors, padsize)),volatile = True).cuda()
    #         predicted = model(padded)
    #         text_probs = F.softmax(predicted).data.cpu().numpy()[0]
    #         for text_idx in range(5):
    #             text_arr[current_text_seq.value*5 + text_idx] = text_probs[text_idx] 
    #         current_text_seq.value += 1

    #       # print(Emotion[np.argmax(F.softmax(predicted).data.cpu().numpy()[0])])
    #       # break
    #     except:
    #       continue
