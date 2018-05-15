import os
import numpy as np
import torch
from torch.utils import data
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle 
	
gt_emotions = './gt_emotions_files/'
vocal_dir = './audio_files_74/'

def make_dataset(mode):
    gt = gt_emotions + mode + '/'
    vocal = vocal_dir + mode + '/'
    gt_files = os.listdir(gt)
    vocal_files = os.listdir(vocal)

    items = []
    for file in sorted(vocal_files):
        gt_path = os.path.join(gt, file)
        items.append(gt_path)
    return items

class gt_mosei(data.Dataset):
    def __init__(self, mode):
        self.items = make_dataset(mode)
        if len(self.items) == 0:
            raise RuntimeError('Found 0 items, please check the data set')

    def __getitem__(self, index):
        gt_path = self.items[index]
        if sys.version_info[0] == 2:
            with open(gt_path,'rb') as f:
                gt_file = pickle.load(f)
        else:
            with open(gt_path,'rb') as f:
                gt_file = pickle.load(f,encoding = 'latin1')

        return gt_file

    def __len__(self):
        return len(self.items)
