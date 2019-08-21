import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import copy
import time

parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str,default='/home/selfdriving/I3D/models32/')
parser.add_argument('-root', type=str, default='/home/selfdriving/mrcnn/bdd12k/')
parser.add_argument('-train_split', type=str, default = '/home/selfdriving/I3D/bdd12k.json')
parser.add_argument('-num_epoch',type=int, default=20)
parser.add_argument('-frame_nb',type=int,  default=32)
parser.add_argument('-class_nb', type=int, default=7 )
parser.add_argument('-resnet_nb', type=int, default=101 )
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-val', default=False)
parser.add_argument('-checkpoints', default=False)

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

from bdd_dataset_32 import BDD_dataset as Dataset
import multiprocessing
multiprocessing.set_start_method('spawn', True) # for vscode debug


def train(args):
    # setup dataset

    transform = transforms.Compose([
        videotransforms.RandomCrop(224)
    ])

    dataset = Dataset(args.train_split, 'train', args.root, args.frame_nb, transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=24, # 24 on jobs
                                             pin_memory=True)
    
    
    val_dataset = Dataset(args.train_split, 'val', args.root, args.frame_nb, transform)
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
    elif args.resnet_nb == 101:
        resnet = torchvision.models.resnet101(pretrained=True)
    elif args.resnet_nb == 152:
        resnet = torchvision.models.resnet152(pretrained=True)
    else:
        raise ValueError('resnet_nb should be in [50|101|152] but got {}'
                         ).format(args.resnet_nb)
    print('load resnet pretrained model')
    # load model
    i3resnet = I3ResNet(copy.deepcopy(resnet), args.frame_nb, args.class_nb, conv_class=True)

    class_weights = [0.4,2,2,2,2,2,1]
    w = torch.FloatTensor(class_weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    optimizer = optim.Adam(i3resnet.parameters(), lr=0.0001, weight_decay=0.001)

    # if args.checkpoints: # nn.Parallel used, further modification needed
    #     checkpoint = torch.load(args.save_model + 'checkpoint.tar')
    #     i3resnet.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']
    #     print('load checkpoint')
    
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
        # if (epoch + 1) % 1 == 0:
        #     torch.save(i3resnet.state_dict(), (save_model + 'net_%d.pth' % (epoch+1)))
            
        if (epoch + 1) % 1 == 0:
            torch.save(i3resnet.state_dict(), (args.save_model + 'net_%d.pth' % (epoch+1)))
            print('checkpoint saved')
            
        if args.val and (epoch + 1)% 10 == 0:
            print("Validation...")
            run_test(val_dataloader, i3resnet, device)

    torch.save(i3resnet.state_dict(), (args.save_model + 'net32frames_Final.pth'))          

def run_test(val_dataloader, i3resnet, device):

    # # Initialize image batch
    # imBatch = Variable(torch.FloatTensor(args.batch_size, 3, args.frame_nb, 224, 224)) # need to adapt to args
    # targetBatch = Variable(torch.FloatTensor(args.batch_size, args.class_nb))

    # # Move network and batch to GPU
    # imBatch = imBatch.to(device)
    # targetBatch = targetBatch.to(device)

    i3resnet.eval()
    AccuracyArr = []
    tic = time.time()
    for i, data in enumerate(val_dataloader):
        if i % 20 == 0:
            print('validation dataset batch:',i)
        # tic = time.time()
        # Read data
        with torch.no_grad():
            img_cpu, label_cpu = data
            img = Variable(img_cpu.to(device))
            label = Variable(label_cpu.to(device))

        pred = i3resnet(img)

        # Calculate accuracy
        predict = torch.sigmoid(pred) > 0.5
        f1 = f1_score(label_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
        AccuracyArr.append(f1)

        torch.cuda.empty_cache()
    toc = time.time()
    print('Time elapsed:',toc-tic )
    print("Validation F1 score:", AccuracyArr[-1])
    i3resnet.train()
    
    
    

if __name__ == '__main__':
    # need to add argparse
    train(args)
