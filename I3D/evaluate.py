import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import f1_score

import numpy as np
import copy
import videotransforms
import time
import argparse

from src.i3res import I3ResNet
from collections import OrderedDict

from bdd_dataset import BDD_dataset as Dataset
# import multiprocessing
# multiprocessing.set_start_method('spawn', True) # for vscode debug

parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-model_path', type=str, default='/home/selfdriving/I3D/models32_interval/net32interval_Final.pth')
parser.add_argument('-root', type=str, default='/home/selfdriving/mrcnn/bdd12k/')
parser.add_argument('-train_split', type=str, default = '/home/selfdriving/I3D/data/bdd12k_4action.json')
parser.add_argument('-frame_nb',type=int,  default=32)
parser.add_argument('-interval',type=int,  default=1)
parser.add_argument('-class_nb', type=int, default=4 )
parser.add_argument('-resnet_nb', type=int, default=101 )
parser.add_argument('-batch_size', type=int, default=4)

args = parser.parse_args()

def eval(args):
    transform = transforms.Compose([
            videotransforms.RandomCrop(224)])


    val_dataset = Dataset(args.train_split, 'val', args.root, args.frame_nb, args.interval, transform)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=24, # on jobs
                                                    pin_memory=True)
    if args.resnet_nb == 50:
        resnet = torchvision.models.resnet50(pretrained=True)
        print('load resnet50 pretrained model...')
    elif args.resnet_nb == 101:
        resnet = torchvision.models.resnet101(pretrained=True)
        print('load resnet101 pretrained model...')
    elif args.resnet_nb == 152:
        resnet = torchvision.models.resnet152(pretrained=True)
        print('load resnet152 pretrained model...')
    else:
        raise ValueError('resnet_nb should be in [50|101|152] but got {}'
                         ).format(args.resnet_nb)

    i3resnet = I3ResNet(copy.deepcopy(resnet), args.frame_nb, args.class_nb, conv_class=True)

    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module'.
        new_state_dict[name] = v

    i3resnet.load_state_dict(new_state_dict)
    print('loaded saved state_dict...')

    i3resnet.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3resnet = i3resnet.to(device)
    # i3resnet = nn.DataParallel(i3resnet)

    AccuracyArr = []
    accuracy = np.zeros((1,args.class_nb))
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            tic = time.time()
            # tic = time.time()
            # Read data
        
            img_cpu, label_cpu = data
            img = Variable(img_cpu.to(device))
            label = Variable(label_cpu.to(device))

            pred = i3resnet(img)

            # Calculate accuracy
            predict = torch.sigmoid(pred) > 0.5
            f1_sample = f1_score(label_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples') # here!!!
            f1 = f1_score(label_cpu.data.numpy(), predict.cpu().data.numpy(), average=None)

            AccuracyArr.append(f1_sample)
            accuracy = np.vstack((accuracy,f1))
            

            if i % 10 == 0:
                toc = time.time()
                print('validation dataset batch:',i)
                print('prediction logits:{}'.format(predict.cpu().data.numpy()))
                print('ground truth:{}'.format(label_cpu.data.numpy()))
                print('f1 score:', f1_sample, 'accumulated f1 score:', np.mean(np.array(AccuracyArr))) #
                print('f1 average:', np.mean(accuracy, axis=0) )
                print('Time elapsed:',toc-tic )
                

            torch.cuda.empty_cache()

    print("Finished Validation")

if __name__ == '__main__':
    # need to add argparse
    eval(args)