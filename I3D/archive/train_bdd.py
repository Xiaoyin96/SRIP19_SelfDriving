import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import sys
import argparse
import copy
import time

parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str,default='models/')
parser.add_argument('-root', type=str, default='/home/selfdriving/mrcnn/bdd12k/')
parser.add_argument('-num_epoch',type=int, default='50')
parser.add_argument('-frame_nb',type=int,  default='64')
parser.add_argument('-class_nb', type=int, default='7' )
parser.add_argument('-resnet_nb', type=int, default='50' )
parser.add_argument('-batch_size', type=int, default='4')
parser.add_argument('-val', default='True')

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import videotransforms

import numpy as np

from src.i3res import I3ResNet

from bdd_dataset import BDD_dataset as Dataset


def train(num_epoch=100, root='/home/selfdriving/mrcnn/bdd12k/', \
        train_split='/home/selfdriving/I3D/data/bdd12k.json', batch_size=4, save_model='models/', \
        frame_nb=64,class_nb=7, resnet_nb=50):
    # setup dataset

    transform = transforms.Compose([
        videotransforms.RandomCrop(224)
    ])

    dataset = Dataset(train_split, 'train', root, transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=16,
                                             pin_memory=True)
    
    if args.val:
        val_dataset = Dataset(train_split, 'val', root, transforms)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=16,
                                                    pin_memory=True)

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    # setup the model
    if args.resnet_nb == 50:
        resnet = torchvision.models.resnet50(pretrained=True)
    elif args.resnet_nb == 101:
        resnet = torchvision.models.resnet101(pretrained=True)
    elif args.resnet_nb == 152:
        resnet = torchvision.models.resnet152(pretrained=True)
    else:
        raise ValueError('resnet_nb should be in [50|101|152] but got {}'
                         ).format(args.resnet_nb)

    i3resnet = I3ResNet(copy.deepcopy(resnet), args.frame_nb, args.class_nb, conv_class=True)
   
    
    # set CPU/GPU devices
    i3resnet.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3resnet = i3resnet.to(device)
    i3resnet = nn.DataParallel(i3resnet) #multiple GPUs

    class_weights = [0.4,2,2,2,2,2,1]
    w = torch.FloatTensor(class_weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    optimizer = optim.Adam(i3resnet.parameters(), lr=0.0001, weight_decay=0.001)


    # train it
    for epoch in range(0, num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch))
        print('-' * 10)

        lossArr = []
        AccuracyArr = []

            # Iterate over data.
        for i, data in enumerate(dataloader):
            tic = time.time()
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs = Variable(inputs.to(device)) #4x3x64x224x224
            labels = Variable(labels.to(device)) #4x7

            optimizer.zero_grad()
            pred = i3resnet(inputs) #4x7

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            loss_cpu = np.array(loss.cpu().data.item())

            lossArr.append(loss_cpu)
            meanLoss = np.mean(np.array(lossArr))

            # Calculate accuracy
            predict = torch.sigmoid(pred) >= 0.5
            f1 = f1_score(labels.cpu().data.numpy(), predict.cpu().data.numpy(), average='samples')
            AccuracyArr.append(f1)


            if i % 10 == 0:
                toc = time.time()
                print('time elapsed', toc - tic)
                #print('prediction:', pred)
                print('prediction logits:{}'.format(predict))

                print('ground truth:{}'.format(labels.cpu().data.numpy()))
                print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                    epoch, i, lossArr[-1], meanLoss))
                print('Epoch %d Iteration %d: F1 %.5f Accumulated F1 %.5f' % (
                    epoch, i, AccuracyArr[-1], np.mean(np.array(AccuracyArr))))

            # if epoch in [int(0.5*num_epoch), int(0.7*num_epoch)] and i==0:
            #     print('The learning rate is being decreased at Iteration %d', i)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] /= 10

        # if i >= args.MaxIteration:
        #     break

        if (epoch + 1) % 5 == 0:
            torch.save(i3resnet.state_dict(), (save_model + 'net_%d.pth' % (epoch + 1)))
        if args.val and (epoch + 1)% 1 == 0:
            print("Validation...")
            run_test(val_dataloader, i3resnet, device)

    torch.save(i3resnet.state_dict(), (save_model + 'net_Final.pth'))          

def run_test(val_dataloader, i3resnet, device):

    i3resnet.eval()
    AccuracyArr = []

    for i, data in enumerate(val_dataloader):
        # tic = time.time()
        # Read data
        img_cpu, label_cpu = data
        img = img.to(device)
        label = label.to(device)

        pred = i3resnet(img)

        # Calculate accuracy
        predict = torch.sigmoid(pred) > 0.5
        f1 = f1_score(label_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
        AccuracyArr.append(f1)


    print("Validation F1 score:", AccuracyArr[-1])
    i3resnet.train()
    
    
    

if __name__ == '__main__':
    # need to add argparse
    train(root=args.root, save_model=args.save_model,frame_nb=args.frame_nb,class_nb=args.class_nb,\
        resnet_nb=args.resnet_nb, num_epoch=args.num_epoch, batch_size=args.batch_size )
