"""----------------------------------------------------Imports-------------------------------------------------------"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
import h5py
from PIL import Image
from sklearn.externals import joblib
import shutil
import os
import random
import pickle
import time
import gc
from graphviz import Digraph
import re
from tensorboardX import SummaryWriter
'----------------------------------------------------Resnet------------------------------------------------------------'
resnet18_filename = 'checkpoint275000_1.pth.tar'
resnet = models.resnet18(pretrained=True)  # Define resnet18 model
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 6).cuda()
state = torch.load(resnet18_filename)
resnet.load_state_dict(state['model'])
modules = list(resnet.children())[:-1]      # delete the last fc layer.
resnet = nn.Sequential(*modules)
'-----------------------------------------------------Vocal Net--------------------------------------------------------'


class VocalNet(nn.Module):
    def __init__(self):
        super(VocalNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=40, stride=1, padding=20)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=40, stride=1, padding=20)

    def forward(self, vocal_input):
        x = F.leaky_relu(F.max_pool1d(self.conv1(vocal_input), 2))
        x = F.leaky_relu(F.max_pool1d(self.conv2(x), 5))
        x = x.view(-1, vocalnet_output_size) # Check this out. View causing problems
        return x


'-----------------------------------------------------CNN Decoder-----------------------------------------------------'


class DecoderCNN(nn.Module):
    def __init__(self, input_size, seq_len, no_of_emotions):
        """Set the hyper-parameters and build the layers."""
        super(DecoderCNN, self).__init__()
        self.bn = nn.BatchNorm1d(seq_len)
        self.linear1 = nn.Linear(input_size, seq_len)
        self.linear2 = nn.Linear(seq_len*seq_len, 6)
        # self.linear3 = nn.Linear(512, 6)
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding = 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=10, stride=3),
        # )
        # self.classifier = nn.Sequential(
        #     # nn.Dropout(),
        #     nn.Linear(128 * 6 * 6, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(4096, no_of_emotions),
        # )

    def forward(self, x):
        """Decode Vocal feature vectors and generates emotions"""
        x = self.bn(x)
        # x = x.unsqueeze(1)
        # x = self.linear1(x)
        # x = self.features(x)
        # x = x.view(x.size(0), 128 * 6 * 6)
        # x = self.classifier(x)
        x = F.relu(self.linear1(x))
        x = x.view(x.size(0), 50*50)
        x = self.linear2(x)
        # x = F.relu(self.linear2(x))

        return x        


'------------------------------------------------------Hyperparameters-------------------------------------------------'
using_vision_network = True
using_vocal_network = False
batch_size = 6
mega_batch_size = 30
no_of_emotions = 6
seq_len = 50
use_CUDA = True
no_of_epochs = 1000
use_pretrained = False
test_mode = False
show_image = True
writer = SummaryWriter('./logs_CREMAD')
read_prev_list = True
'-----------------------------------Parameters NOT subject to change---------------------------------------------------'

len_waveform = 320  # This is the length of a 1 frame long waveform vector
vocalnet_output_size = 1280  # VocalNet outputs a 1280X1 feature vector
resnet18_output_size = 512  # Resnet Outputs a 1X512X1X1 feature vector.
if using_vision_network and using_vocal_network:
    decoder_input_size = vocalnet_output_size+resnet18_output_size
elif using_vision_network:
    decoder_input_size = resnet18_output_size
else:
    decoder_input_size = vocalnet_output_size 

Vocal_encoder = VocalNet()  # Define the vocalnet model
cnn_decoder = DecoderCNN(input_size=decoder_input_size, seq_len = seq_len, no_of_emotions = no_of_emotions)  # Define the shared LSTM Decoder.
# res = cnn_decoder(torch.autograd.Variable(torch.FloatTensor(150,1,1792).cuda(),requires_grad = True))
# writer.add_graph(cnn_decoder, res)

curr_epoch = 0
total = 0

resnet = resnet.cuda()
Vocal_encoder = Vocal_encoder.cuda()
cnn_decoder = cnn_decoder.cuda()
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.CrossEntropyLoss()
params =  list(cnn_decoder.parameters())+list(resnet.parameters()) #+ list(Vocal_encoder.parameters()) 
print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, 0.0001)

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='resnet18_vocalnet_LSTM'):
    filename = filename +'_'+str(state['epoch'])+'.pth.tar' 
    torch.save(state, filename)
    if is_final:
        shutil.copyfile(filename, 'model_final.pth.tar')


'-------------------------------------------Setting into train mode----------------------------------------------------'

if not test_mode:
    cnn_decoder.train(True)
    Vocal_encoder.train(True)
    resnet.train(True)
else:
    cnn_decoder.train(False)
    Vocal_encoder.train(False)
    resnet.train(False)
'----------------------------------------------------------------------------------------------------------------------'

combined_seq_total = ""
target_seq_total = ""
directory = "./all_sequences/train_30"
prev_loss = 0

# Will contain all of the sequences
sequences = {}
for idx,files in enumerate(os.listdir(directory)):
    sequences.update({idx:files})
# you can't shuffle a dictionary, but what you can do is shuffle a list of its keys
keys = list(sequences.keys())

# Pick whatver loop you want

# vocal_seq = np.random.randn(batch_size,seq_len,1,len_waveform).astype(np.float32)
# vision_seq = np.random.randn(batch_size,seq_len,resnet18_output_size,1,1).astype(np.float32)
# target_seq = np.random.randn(batch_size,2000).astype(np.float32)


if not read_prev_list:
    random.shuffle(keys)
    input_list = [(key, sequences[key]) for key in keys]
    print('Size of Training Set = ' + str(len(input_list)))
    # input_list = input_list[:4]
    with open("input_list.pickle", 'wb') as g:
        pickle.dump(input_list, g)
else:
    with open("input_list.pickle", 'rb') as f: # Check this out 
        input_list = pickle.load(f)
for epoch in range(curr_epoch, no_of_epochs):  
    j_start = 0
    running_loss = 0
    running_corrects = 0
    if use_pretrained:
        checkpoint = torch.load('resnet18_vocalnet_LSTM_100_0.pth.tar')
        cnn_decoder.load_state_dict(checkpoint['cnn_decoder'])
        Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
        resnet.load_state_dict(checkpoint['resnet18'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        j_start = checkpoint['j_start']
        running_loss = checkpoint['loss']
        running_corrects = checkpoint['correct']
        curr_epoch = checkpoint['epoch']
        use_pretrained = False

    input_list = input_list[j_start:]
    K= 0 
    for j in range(j_start, len(input_list), batch_size):

    
        if ((len(sequences) - j) > batch_size):
            input_batch = input_list[j:j+batch_size]
        else:
            break

        for batch in range(batch_size):
            with open(directory+"/"+str(input_batch[0][1]), 'rb') as f: # Check this out 
                data = pickle.load(f)
            # print(data[0])
            target_numpy = [data[0]]
            # print(len(data[1]))
            # print(len(data[1][0]))
            vocal_seq_numpy = np.array(data[1], dtype = np.float32)
            vision_seq_numpy = data[2]
            # print(vocal_seq_numpy[6][200])
            vision_seq_input = np.empty((seq_len,3,224,224), dtype=np.float32)
            vocal_seq_input = np.empty((seq_len, 1,320), dtype = np.float32)

            for seq in range(seq_len):
                file_name = vision_seq_numpy[seq]
                img = Image.open(file_name)
                pixels = np.array(img,dtype = np.uint8)/255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                pixels = std * pixels + mean
                if show_image:
                    plt.imshow(pixels, interpolation='nearest')
                    plt.show()
                    show_image = False
                pixels = pixels.transpose(2, 0 ,1)
                vision_seq_input[seq,:,:,:] = pixels 
                vocal_seq_input[seq,:,:] = vocal_seq_numpy[seq]

            vision_seq_i = Variable(torch.Tensor(vision_seq_input).cuda()) # Check whether same as input or not
            vision_seq_o =resnet(vision_seq_i) # Check this out
            vision_seq_o = torch.squeeze(vision_seq_o)
            target_seq_o = Variable(torch.LongTensor(target_numpy)).cuda()   # Check this out. Torch Tensor v/s Numpy
            # print(target_seq_o)
            vocal_seq_i = Variable(torch.from_numpy(vocal_seq_input)).cuda() # Check this out. Torch Tensor v/s Numpy
            # print(vocal_seq_i[6][0][200]) # Check whether same as input or not 
            vocal_seq_o = Vocal_encoder(vocal_seq_i)
            # print(vocal_seq_o.size())
            # vocal_seq_o = torch.transpose(vocal_seq_o,0,1)

            # print(vision_seq_o.size(),vocal_seq_o.size())

            if using_vision_network and using_vocal_network:
                combined_seq_i = torch.cat((vocal_seq_o, vision_seq_o), 1).cuda()

            elif using_vision_network and not using_vocal_network:
                combined_seq_i = vision_seq_o
            else:
                combined_seq_i = vocal_seq_o

            # target_seq_o = target_seq_o.unsqueeze(0)
            combined_seq_i = combined_seq_i.unsqueeze(0)

            if batch == 0:
                combined_seq_total = combined_seq_i
                target_seq_total = target_seq_o
            else:
                combined_seq_total = torch.cat((combined_seq_total, combined_seq_i), 0)
                target_seq_total = torch.cat((target_seq_total, target_seq_o), 0)

        lstm_output = cnn_decoder(combined_seq_total)
        # print(lstm_output)
        # print(target_seq_total)
        loss = criterion(lstm_output, target_seq_total)
        writer.add_histogram("target_seq_epoch_"+str(epoch+1),target_seq_o,j+batch_size)
        writer.add_histogram("lstm_output_epoch_"+str(epoch+1),lstm_output,j+batch_size)
        writer.add_histogram("resnet_output_epoch_"+str(epoch+1), vision_seq_o, j+batch_size)
        # writer.add_histogram("resnet_input_epoch_"+str(epoch+1), vision_seq_i4, j+batch_size)

        # print(lstm_output)
        # print(target_seq_total)
        # if j%mega_batch_size==0:
        optimizer.zero_grad()
        cnn_decoder.zero_grad()
        Vocal_encoder.zero_grad()

        loss.backward()
        # if (j+batch_size)%mega_batch_size==0:
        optimizer.step()
        _, preds = torch.max(lstm_output.data, 1)
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == target_seq_total.data)
        if j%12 == 0:
            print(list(resnet.parameters())[-1].grad[:])
        #     print(list(resnet.parameters())[-1][0])
        #     print(list(cnn_decoder.parameters())[-1].grad[:5])
        #     print(list(cnn_decoder.parameters())[-1][0])
        #     print(list(Vocal_encoder.parameters())[-1].grad)
        #     print(list(cnn_decoder.parameters()))
        #     lstm_vector_output = lstm_output.view((seq_len, no_of_emofeatures))[seq_len - 1].data.cpu().numpy()
        #     target_vector_output = target_seq_total.view((seq_len, no_of_emofeatures))[seq_len - 1].data.cpu().numpy()

        # for idx,ele in enumerate(Vocal_encoder.parameters()):
        #     writer.add_histogram("VocalNet_"+ str(idx)+"_epoch_"+str(epoch+1), ele.grad,j+batch_size)
        for idx,ele in enumerate(cnn_decoder.parameters()):
            writer.add_histogram("Decoder_"+str(idx)+"_epoch_"+str(epoch+1), ele.grad,j+batch_size)
        for idx,ele in enumerate(resnet.parameters()):
            writer.add_histogram("Resnet_"+ str(idx)+"_epoch_"+str(epoch+1), ele.grad,j+batch_size)        
        # if (j+batch_size)%mega_batch_size==0:
        running_accuracy = 100.0*float(running_corrects)/float(j+batch_size)
        K+=1
        average_loss = float(running_loss)/float(K)
        writer.add_scalar("Accuracy_epoch_"+str(epoch+1), running_accuracy, j+batch_size)
        writer.add_scalar("Average_loss_epoch_"+str(epoch+1), average_loss, j+batch_size)
        print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f, Accuracy: %.4f'
            % (epoch+1, j+batch_size, average_loss, running_accuracy))

        if (j+batch_size)%100 == 0:
            save_checkpoint({
                'epoch': 0,
                'accuracy': running_accuracy,
                'loss' : running_loss,
                'correct' : running_corrects,
                'j_start' : j+batch_size,
                'cnn_decoder': cnn_decoder.state_dict(),
                'Vocal_encoder': Vocal_encoder.state_dict(),
                'resnet18': resnet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, 'resnet18_vocalnet_LSTM_'+str(j+batch_size))
        # file = open("Monitor_Bimodal.txt", "a")
        # file.write('Training -- Epoch [%d], Sample [%d], Low Pass Loss: %.4f, Batch Loss: %.4f'
        #   % (epoch+1, j, curr_loss, loss.data[0]))
        # file.write("\n")
        # file.close()
    '-------------------------------------------------Saving model after every epoch-----------------------------------'
    if not test_mode:
        save_checkpoint({
            'epoch': 0,
            'accuracy': running_accuracy,
            'loss' : running_loss,
            'correct' : running_corrects,
            'j_start' : 0,
            'cnn_decoder': cnn_decoder.state_dict(),
            'Vocal_encoder': Vocal_encoder.state_dict(),
            'resnet18': resnet.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False)
'------------------------------------------------------Saving model after training completion--------------------------'
if not test_mode:
    save_checkpoint({
        'epoch': 0,
        'accuracy': running_accuracy,
        'loss' : running_loss,
        'correct' : running_corrects,
        'j_start' : 0,
        'cnn_decoder': cnn_decoder.state_dict(),
        'Vocal_encoder': Vocal_encoder.state_dict(),
        'resnet18': resnet.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, False)

    