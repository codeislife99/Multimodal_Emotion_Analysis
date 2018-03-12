import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import shutil
from PIL import Image
from matplotlib.pyplot import imshow 


num_classes = 6
confusion_mat = np.array([[0 for x in range(num_classes)] for y in range(num_classes)],np.int32) 
batch_size = 8
resnet18 = models.resnet18(pretrained=True)
freeze_net = True
if freeze_net:
    for p in resnet18.parameters():
        p.requires_grad = False

resnet18.fc=nn.Linear(512,num_classes)

# print(resnet18)
data_transforms = {
    'train': transforms.Compose([
        # transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './custom'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

print(class_names)
use_gpu = torch.cuda.is_available()
use_pretrained = True

def save_checkpoint(state, is_best,filename='checkpoint'):
    filename = filename +'_'+str(state['epoch'])+'.pth.tar' 
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    # inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.2)  # pause a bit so that plots are updated

def test_image(model,path):
    model.train(False)  
    img = Image.open(path)
    img = img.resize((224,224),Image.ANTIALIAS)

    img = np.array(img,dtype = np.float32)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    imshow(img)
    img = img.transpose(2,0,1)
    inputs = np.empty([1,3,224,224], dtype = np.float32)
    inputs[0,:,:,:] = img
    inputs= Variable(torch.from_numpy(inputs).float())
    outputs = model(inputs)
    _, pred = torch.max(outputs.data, 1)
    print(pred)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    curr_epoch = 0
    if use_pretrained:
        state = torch.load('checkpoint275000_1.pth.tar')
        curr_epoch = state['epoch']
        model.load_state_dict(state['model'])
        # optimizer.load_state_dict(state['optimizer'])
        best_acc = state['best_ac']

    for epoch in range(curr_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        confusion_mat = [[0 for x in range(num_classes)] for y in range(num_classes)] 

        # Each epoch has a training and val phase, skipping val
        for phase in ['train','val']:
            # print(dataset_sizes['test'])
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            K = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                print(inputs.requires_grad)
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                print(labels.size())
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # index1=outputs.data.numpy();
                # index2=preds.data.numpy();

                # type(outputs)
                # print type(labels.data[0])
                # print type(preds[0])

                # if phase == 'val':
                #     confusion_mat[labels.data[0]][preds[0]]+=1



                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                K+=1
                running_accuracy = float(running_corrects)/float(K*batch_size)

                print(K*batch_size, float(running_loss)/float(K), 100*running_accuracy)

                if (K*batch_size%25000==0) and (phase == 'train'):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'best_ac': best_acc,
                        'optimizer' : optimizer.state_dict(),
                        'model_ac' : 100.0*running_accuracy
                    }, False,'checkpoint'+str(K*batch_size))                     

                # print(running_corrects)
                # print(labels.data)
                # print(dataset_sizes[phase])
                # print(float(running_corrects) / float(dataset_sizes[phase]))
                # if phase=='test':
                #     # out = torchvision.utils.make_grid(inputs[0])
                #     # imshow(out)
                #     print(labels[0])
                #     print(preds[0])
                #     print(labels[1])
                #     print(preds[1])
                #     print(labels[2])
                #     print(preds[2])
                #     print(labels[3])
                #     print(preds[3])
                #     print(labels[4])
                #     print(preds[4])
                #     print(labels[5])
                #     print(preds[5])
                #     print(labels[6])
                #     print(preds[6])
                    # print(labels[7])
                    # print(preds[7])

            epoch_loss = running_loss*batch_size / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / float(dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}%'.format(
                phase, epoch_loss, 100.0*epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # Save the Model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'best_ac': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, True)

        print()

        # Save the Model
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_ac': best_acc,
            'optimizer' : optimizer.state_dict(),
            'model_ac' : 100.0*epoch_acc
        }, False)
        # path='laundromat.jpg'
        # test_image(model,path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




if use_gpu:
	resnet18 = resnet18.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
if freeze_net:
    optimizer_ft = optim.Adam(resnet18.fc.parameters(), lr=0.01)
else:
    optimizer_ft = optim.Adam(resnet18.parameters(), lr=0.01)    

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


resnet18 = train_model(resnet18, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)