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
import json

from src.i3res_new import I3ResNet
from collections import OrderedDict

from bdd_dataset_feature import BDD_dataset as Dataset

parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-model_path', type=str, default='/home/selfdriving/I3D/models_resnet101woside/net_Final.pth')
parser.add_argument('-save_path', type=str, default='/home/selfdriving/I3D/features_val/')
parser.add_argument('-root', type=str, default='/home/selfdriving/mrcnn/bdd12k/')
parser.add_argument('-train_split', type=str, default = '/home/selfdriving/I3D/data/4action_reason.json')
parser.add_argument('-frame_nb',type=int,  default=32)
parser.add_argument('-interval',type=int,  default=1)
parser.add_argument('-class_nb', type=int, default=4)
parser.add_argument('-resnet_nb', type=int, default=101)
parser.add_argument('-batch_size', type=int, default=1)

args = parser.parse_args()

class Hook():
    '''
    A simple hook class that returns the input and output of a layer during forward/backward pass
    '''
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def extract_feature(args):

    transform = transforms.Compose([
            videotransforms.RandomCrop(224)])
    dataset = Dataset(args.train_split, 'val', args.root, args.frame_nb, args.interval, transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=24, # 24 on jobs
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

    i3resnet = I3ResNet(copy.deepcopy(resnet), args.frame_nb, args.class_nb, side_task=False, conv_class=True)
    # print(i3resnet.layer3[0].downsample[1])
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

    with torch.no_grad():
        hook = Hook(i3resnet.layer3)
        print('registered Hook...')
        for i, data in enumerate(dataloader):
            vid, img_cpu, action_cpu, reson_cpu = data
            img = Variable(img_cpu.to(device))
            action = Variable(action_cpu.to(device))
            pred = i3resnet(img)
            feature = hook.output.cpu().data.numpy()
            feature = np.squeeze(feature)
            np.save((args.save_path + '{}.npy'.format(vid[0])),feature)
            
            print('Saved feature numbers:', i+1)
    print('finished extracting features')

if __name__ == '__main__':
    extract_feature(args)
    