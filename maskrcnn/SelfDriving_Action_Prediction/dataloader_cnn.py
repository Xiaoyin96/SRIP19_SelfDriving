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
from scipy.ndimage import zoom
import json
import random


class BatchLoader(Dataset):
    def __init__(self, imageRoot, gtRoot, reasonRoot=None, batchSize=1, cropSize=(1280, 720)):
        super(BatchLoader, self).__init__()

        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.cropSize = cropSize
        self.batchsize = batchSize

        with open(gtRoot) as json_file:
            data = json.load(json_file)
        with open(reasonRoot) as json_file:
            reason = json.load(json_file)
        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name']) 

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            self.imgNames.append(osp.join(self.imageRoot, img['file_name']))
            self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
            self.reasons.append(torch.LongTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))
        self.perm = list(range(self.count))
        random.shuffle(self.perm)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[self.perm[ind]]
        target = np.array(self.targets[self.perm[ind]], dtype=np.int64)
        reason = np.array(self.reasons[self.perm[ind]], dtype=np.int64)

        img = Image.open(imgName)

        transform = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
        ])
        img_ = transform(img)

        batchDict = {
            'img': img_,
            'target': target,
            'ori_img': np.array(img),
            'reason': reason
        }
        return batchDict


