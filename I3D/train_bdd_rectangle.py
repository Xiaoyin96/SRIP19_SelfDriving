import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import copy
import time

parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str,default='/home/selfdriving/I3D/model/models_640x360/')
parser.add_argument('-root', type=str, default='/home/selfdriving/mrcnn/bdd12k/')
parser.add_argument('-train_split', type=str, default = '/home/selfdriving/I3D/data/4action_reason.json')
parser.add_argument('-num_epoch',type=int, default=40)
parser.add_argument('-frame_nb',type=int,  default=16)
parser.add_argument('-interval',type=int,  default=1)
parser.add_argument('-class_nb', type=int, default=4 )
parser.add_argument('-resnet_nb', type=int, default=101)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-val', default=True)
parser.add_argument('-checkpoints', default=False)
parser.add_argument('-side_task', default=False)
parser.add_argument('-reason_nb', type=int, default=21)

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

from src.i3res_rectangle import I3ResNet

from bdd_dataset_rectangle import BDD_dataset as Dataset
import multiprocessing
multiprocessing.set_start_method('spawn', True) # for vscode debug


def train(args):
    # setup dataset

    # transform = transforms.Compose([
    #     videotransforms.RandomCrop(224)
    # ])

    dataset = Dataset(args.train_split, 'train', args.root, args.frame_nb, args.interval)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=24, # 24 on jobs
                                             pin_memory=True)
    
    
    val_dataset = Dataset(args.train_split, 'val', args.root, args.frame_nb, args.interval)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=24,
                                                pin_memory=True)

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    # setup the model
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
    
    # load model
    i3resnet = I3ResNet(copy.deepcopy(resnet), args.frame_nb, args.class_nb, args.reason_nb, args.side_task, conv_class=True)

    class_weights = [1,2,2,2]
    w = torch.FloatTensor(class_weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()

    if args.side_task:
        criterion2 = nn.BCEWithLogitsLoss().cuda()

    optimizer = optim.Adam(i3resnet.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # set CPU/GPU devices
    i3resnet.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3resnet = i3resnet.to(device)
    i3resnet = nn.DataParallel(i3resnet)

    # train it
    for epoch in range(0, args.num_epoch):
        print('Epoch {}/{}'.format(epoch, args.num_epoch))
        print('-' * 10)

        lossArr = []
        AccuracyArr = []
        ReasonArr = []

            # Iterate over data.
        for i, data in enumerate(dataloader):
            tic = time.time()
            # get the inputs
            inputs, action, reason = data

            # wrap them in Variable
            inputs = Variable(inputs.to(device)) #4x3x64x224x224
            action = Variable(action.to(device)) #4x4
            reason = Variable(reason.to(device)) # 4x21


            optimizer.zero_grad()
            if args.side_task:
                pred, pred_reason = i3resnet(inputs)
                loss1 = criterion(pred, action)
                loss2 = criterion2(pred_reason, reason)
                loss = loss1 + loss2 # joint loss
            else: 
                pred = i3resnet(inputs) #4x4
                loss = criterion(pred, action)
            
            loss.backward()
            optimizer.step()
            loss_cpu = np.array(loss.cpu().data.item())

            lossArr.append(loss_cpu)
            meanLoss = np.mean(np.array(lossArr))

            # Calculate accuracy
            predict = torch.sigmoid(pred) >= 0.5
            f1 = f1_score(action.cpu().data.numpy(), predict.cpu().data.numpy(), average='samples')
            AccuracyArr.append(f1)    
            if args.side_task:
                predict_reason = torch.sigmoid(pred_reason) >= 0.5
                f1_reason = f1_score(reason.cpu().data.numpy(), predict_reason.cpu().data.numpy(), average='samples')
                ReasonArr.append(f1_reason)

            if i % 10 == 0:
                toc = time.time()
                print('time elapsed', toc - tic)
                #print('prediction:', pred)
                print('prediction logits:{}'.format(predict))

                print('ground truth:{}'.format(action.cpu().data.numpy()))
                print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                    epoch, i, lossArr[-1], meanLoss))
                print('Epoch %d Iteration %d: F1 %.5f Accumulated F1 %.5f' % (
                    epoch, i, AccuracyArr[-1], np.mean(np.array(AccuracyArr))))
                # print('Epoch %d Iteration %d: F1 %.5f Accumulated Reason F1 %.5f' % (
                #     epoch, i, ReasonArr[-1], np.mean(np.array(ReasonArr))))
                           
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(i3resnet.state_dict(), (args.save_model + 'net_%d.pth' % (epoch+1)))
            print('checkpoint saved')   

    torch.save(i3resnet.state_dict(), (args.save_model + 'net_Final.pth'))   
    print('Finished Training','-'*10)
    if args.val:
        print("Validation...")
        run_test(args, val_dataloader, i3resnet, device)       

def run_test(args, val_dataloader, i3resnet, device):

    AccuracyArr = []
    accuracy = np.zeros((1,args.class_nb))
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
           
            # Read data
            img_cpu, action_cpu, reason_cpu = data
            img = Variable(img_cpu.to(device))
            action = Variable(action_cpu.to(device))
            reason = Variable(reason_cpu.to(device))

            if args.side_task:
                pred, pred_reason = i3resnet(img)
            else:
                pred = i3resnet(img)

            # Calculate accuracy
            predict = torch.sigmoid(pred) > 0.5
            f1_sample = f1_score(action_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples') # here!!!
            f1 = f1_score(action_cpu.data.numpy(), predict.cpu().data.numpy(), average=None)

            AccuracyArr.append(f1_sample)
            accuracy = np.vstack((accuracy,f1))

            
            if i % 100 == 0:
                print('validation dataset batch:',i)
                print('prediction logits:{}'.format(predict.cpu().data.numpy()))
                print('ground truth:{}'.format(action_cpu.data.numpy()))
                print('f1 score:', f1_sample, 'accumulated f1 score:', np.mean(np.array(AccuracyArr))) #
                print('f1 average:', np.mean(accuracy, axis=0))
                

            torch.cuda.empty_cache()

    print("Finished Validation")
    i3resnet.train()
    
    
    

if __name__ == '__main__':
    # need to add argparse
    train(args)
