
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from PIL import Image
import json
import random

class FeatureLoader(Dataset):
    def __init__(self, featureRoot, actionRoot):
        super(FeatureLoader, self).__init__()
        
        self.featureRoot = featureRoot
        self.actionRoot = actionRoot
        # self.bboxRoot = bboxRoot
        
        dirs = os.listdir(featureRoot)
        self.vidlist = [x.split('.')[0] for x in dirs] # video name list
        with open(actionRoot, 'r') as f:
            label = json.load(f)
        self.label = label
        # self.box_dict = torch.load(bboxRoot) #  key is vid name, value is a in-order BoxList
        
    def __len__(self):
        return len(self.vidlist)
    
    def __getitem__(self, idx):
        # get features and bboxlist according to each video name
        batchDict = {}
        vid = self.vidlist[idx]
        features = np.load(os.path.join(self.featureRoot, vid+'.npy'))
        features = features.transpose([1,0,2,3]) # nf x 1024 x 14 x 14
        feature = features[-1,:,:,:] # last frame's features
        feature = torch.Tensor(feature)
        action = np.asarray(self.label[vid]['actions'], np.float32) #size: num_cls x 1
        # bbox = self.box_dict[vid] # a list of nf BoxList
        batchDict = {
                'vid': vid,
                'feature': feature,
                'action': action
                # 'bbox': bbox
                }
        return batchDict




