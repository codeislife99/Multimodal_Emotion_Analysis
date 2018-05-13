import os
import numpy as np
import torch
from torch.utils import data
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle 

vision_dir = './vision_files/'
vocal_dir = './audio_files_74/'
gt_emotions = './gt_emotions_files/'
emb_dir = './text_files_segbased/'
emb2_dir = './text_files_videobased/'
def make_dataset(mode, segment = True):
    vision = vision_dir + mode+ '/'
    vocal = vocal_dir + mode + '/'
    gt = gt_emotions + mode + '/'
    if segment:
        emb = emb_dir + mode + '/'
    else:
        emb = emb2_dir + mode + '/'
    vision_files = os.listdir(vision)
    vocal_files = os.listdir(vocal)
    gt_files = os.listdir(gt)
    emb_files = os.listdir(emb)
    items = []
    for file in sorted(vocal_files):
        vision_path = os.path.join(vision, file)
        vocal_path = os.path.join(vocal, file)
        gt_path = os.path.join(gt, file)
        emb_path = os.path.join(emb, file)
        items.append((vision_path,vocal_path,gt_path,emb_path))
    return items

class mosei(data.Dataset):
    def __init__(self, mode,segment = True):
        self.items = make_dataset(mode,segment)
        if len(self.items) == 0:
            raise RuntimeError('Found 0 items, please check the data set')

    def __getitem__(self, index):
        vision_path, vocal_path, gt_path, emb_path = self.items[index]
        print(emb_path)
        if sys.version_info[0] == 2:
            with open(vision_path,'rb') as f:
                vision_file = pickle.load(f)
            with open(vocal_path,'rb') as f:
                vocal_file = pickle.load(f)
            with open(gt_path,'rb') as f:
                gt_file = pickle.load(f)
            with open(emb_path,'rb') as f:
                emb_file = pickle.load(f)
        else:
            with open(vision_path,'rb') as f:
                vision_file = pickle.load(f,encoding = 'latin1')
            with open(vocal_path,'rb') as f:
                vocal_file = pickle.load(f,encoding = 'latin1')
            with open(gt_path,'rb') as f:
                gt_file = pickle.load(f,encoding = 'latin1')
            with open(emb_path,'rb') as f:
                try:
                    emb_file = pickle.load(f,encoding = 'latin1')
                except:
                    emb_file = pickle.load(f,encoding = 'bytes')

        return vision_file, vocal_file, gt_file,emb_file, vision_path, vocal_path, gt_path, emb_path

    def __len__(self):
        return len(self.items)