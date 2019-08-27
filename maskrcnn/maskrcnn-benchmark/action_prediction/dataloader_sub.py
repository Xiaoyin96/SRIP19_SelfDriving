# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:16:36 2019

@author: epyir
"""
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image
import json
import random


class BatchLoader(Dataset):
    def __init__(self, imageRoot, gtRoot, batchSize=1, cropSize=(1280, 720)):
        super(BatchLoader, self).__init__()

        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.cropSize = cropSize
        self.batchsize = batchSize

        with open(gtRoot) as json_file:
            data = json.load(json_file)

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets = [], []
        forward_sample = 0
        for i, img in enumerate(imgNames):
            if int(action_annotations[i]['category_id']) == 0 or int(action_annotations[i]['category_id']) == 1:
#            if not (int(action_annotations[i]['category_id']) == 0 and forward_sample > 1000):
                f = str(int(img['id']) - 1) + '.npy'
                self.imgNames.append(osp.join(self.imageRoot, f))
                self.targets.append(int(action_annotations[i]['category_id']))
#                forward_sample += 1

        self.count = len(self.imgNames)
        self.perm = list(range(self.count))
        random.shuffle(self.perm)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[self.perm[ind]]
        target = np.array(self.targets[self.perm[ind]], dtype=np.int64)
        img = np.load(imgName)

        # transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#          transforms.ToTensor(),
         # transforms.Normalize(
         #  mean=[0.485, 0.456, 0.406],
         #  std=[0.229, 0.224, 0.225])
        # ])
        
        img = torch.Tensor(img)
        # target = torch.LongTensor(target)
        #img = np.array(img)
        #img = np.transpose(img, (2, 0, 1))
        batchDict = {
            'img': img,
            'target': target,
        }
        return batchDict



