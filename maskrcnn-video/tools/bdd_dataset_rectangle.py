import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2
from PIL import Image

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2])) # convert numpy to tensor


def load_rgb_frames(image_dir, split, vid, num_frames, interval):
    frames = []
    for i in range(1, num_frames+1, interval):
        img = cv2.imread(os.path.join(image_dir, split, 'img/', vid, vid+'_'+str(i).zfill(2)+'.jpg'))[:, :, [2, 1, 0]] 
        w, h, c = img.shape
        dim = (int(w/2), int(h/2))
        img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR) # resize to 640 x 360 x 3 
        # print(img.shape)
        img = (img/255.)*2 - 1 # normalize
        frames.append(img) 

    return np.asarray(frames, dtype=np.float32) # a list of np.array (640,360,3)


def make_dataset(split_file, split, root):
    
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():  
        if data[vid]['subset'] != split:
            continue
        if not os.path.exists(os.path.join(root, split, 'img/' + vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, split, 'img/' + vid)))         
        if num_frames < 64:
            continue

        action = np.asarray(data[vid]['actions'], np.float32) #size: num_cls x 1
        reason = np.asarray(data[vid]['reason'], np.float32) # size: 21 x 1

        
        dataset.append((vid, action, reason, num_frames))
        i += 1
        if i%1000 == 0:
            print('load data:', i)
    
    return dataset


class BDD_dataset(data_utl.Dataset):

    def __init__(self, split_file, split, root, frame_nb, interval, transforms=None):
        
        self.data = make_dataset(split_file, split, root=root)
        self.split_file = split_file
        self.transforms = transforms
        self.root = root
        self.split = split
        self.frame_nb = frame_nb
        self.interval = interval

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, action, reason, nf = self.data[index]  
        imgs = load_rgb_frames(self.root, self.split, vid, nf, self.interval) # size: nf x 256 x 256 x 3
        num = len(imgs)
        if num < self.frame_nb:
            raise ValueError('frame number{} is smaller than assigned frame_nb{}'
                         ).format(num, self.frame_nb)
        imgs = imgs[-self.frame_nb:,:,:,:] # last nf frames
        # imgs = self.transforms(imgs) # nf x 224 x 224 x 3
         
        batchDict = {
                'vid': vid,
                'imgs': video_to_tensor(imgs),
                'action': torch.from_numpy(action),
                'reason': torch.from_numpy(reason)
                # 'bbox': bbox
                }
        return batchDict
        # return video_to_tensor(imgs), torch.from_numpy(action), torch.from_numpy(reason) # convert to tensor: 3 x nf x 224 x 224

    def __len__(self):
        return len(self.data)
