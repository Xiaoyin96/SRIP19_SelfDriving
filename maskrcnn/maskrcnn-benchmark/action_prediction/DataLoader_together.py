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

from maskrcnn_benchmark.data.transforms import transforms as T


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
            if action_annotations[ind]['category'][4] == 0:
                self.imgNames.append(osp.join(self.imageRoot, img['file_name']))
                self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                self.reasons.append(torch.LongTensor(reason[i]['reason']))

        '''# get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets = [], []
        for i, img in enumerate(imgNames):
            if action_annotations[i]['category'][4] == 0:
                self.imgNames.append(osp.join(self.imageRoot, img['file_name']))
                self.targets.append(torch.LongTensor(action_annotations[i]['category']))'''

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))
        self.perm = list(range(self.count))
        random.shuffle(self.perm)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        # test = True
        imgName = self.imgNames[self.perm[ind]]
        target = np.array(self.targets[self.perm[ind]], dtype=np.int64)
        reason = np.array(self.reasons[self.perm[ind]], dtype=np.int64)
        # target = one_hot(target, 4)

        img_ = Image.open(imgName)

        color_jitter = T.ColorJitter(
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
            )
        normalize_transform = T.Normalize(
                mean=[102.9801, 115.9465, 122.7717],
                std=[1., 1., 1.],
                to_bgr255=True,
            )
        transform = T.Compose(
                [color_jitter,
                 #T.Resize(self.cropSize[1], self.cropSize[0]),
                 T.ToTensor(),
                 normalize_transform,
                ]
            )
        img, target = transform(img_, target)
        batchDict = {
                'img': img,
                'target': torch.FloatTensor(target)[0:4],
                'ori_img': np.array(img_),
                'reason': torch.FloatTensor(reason)
            }
        return batchDict


# def toDivisibility(img, pad=(0,0,8,8)):
#     m = nn.ConstantPad2d(pad, 0)
#     return m(img)

# def make_one_hot(labels, C=4):
#     '''
#     Converts an integer label torch.autograd.Variable to a one-hot Variable.
#
#     Parameters
#     ----------
#     labels : torch.autograd.Variable of torch.cuda.LongTensor
#         N x 1 x H x W, where N is batch size.
#         Each value is an integer representing correct classification.
#     C : integer.
#         number of classes in labels.
#
#     Returns
#     -------
#     target : torch.autograd.Variable of torch.cuda.FloatTensor
#         N x C x H x W, where C is class number. One-hot encoded.
#     '''
#     one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
#     target = one_hot.scatter_(1, labels.data, 1)
#
#     target = Variable(target)
#
#     return target

# def one_hot(label, C=4):
#     one_hot_t = torch.LongTensor(1, C) % 3
#     one_hot_t = one_hot_t.zero_()
#     if label == 0:
#         one_hot_t[0, 0] = 1
#     elif label == 1:
#         one_hot_t[0, 1] = 1
#     elif label == 2:
#         one_hot_t[0, 2] = 1
#     elif label == 3:
#         one_hot_t[0, 3] = 1
#
#     return one_hot_t

