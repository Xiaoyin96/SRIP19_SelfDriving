# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:08:43 2019

@author: epyir
"""
import argparse
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import normalize

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.structures.image_list import to_image_list
from baseline_sub import baseline
from dataloader_sub import BatchLoader
from test2 import test

import torch.optim as optim


def train(cfg, args):

    device = torch.device(cfg.MODEL.DEVICE)
    outdir = cfg.OUTPUT_DIR

    '''def collate_fn_padd(batch):
        print(batch)
        lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
        batch = [ torch.Tensor(t).to(device) for t in batch ]
        batch = torch.nn.utils.rnn.pad_sequence(batch)
        mask = (batch != 0).to(device)
        return new_batch, lengths, mask'''

    # Initialize the network
    model = baseline(cfg, is_cat=args.is_cat)
    class_weights = [1, 1, 5, 5]  # could be adjusted
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize optimizer
    #optimizer = optim.SGD(model.parameters(), lr=float(args.initLR), momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=float(args.initLR), weight_decay=args.weight_decay)

    # Initialize image batch
    # imBatch = Variable(torch.FloatTensor(args.batch_size, 1024, 14, 14))
    targetBatch = Variable(torch.LongTensor(args.batch_size))

    # Move network and batch to gpu
    # imBatch = imBatch.cuda(device)
    targetBatch = targetBatch.cuda(device)
    model = model.cuda(device)
    print(model)

    # Initialize dataloader
    Dataset = BatchLoader(
        imageRoot=args.imageroot,
        gtRoot=args.gtroot,
        #cropSize=(args.imWidth, args.imHeight)
    )
    #dataloader = DataLoader(Dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn_padd)
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    lossArr = []
    AccuracyArr = []
    accuracy = 0
    iteration = 0

    for epoch in range(0, 100):
        trainingLog = open(outdir + ('trainingLog_{0}.txt'.format(epoch)), 'w')
        accuracy = 0
        trainingLog.write(str(args))
        for i, dataBatch in enumerate(dataloader):
            iteration = i + 1
            #print(dataBatch)

            # Read data, under construction
            img_cpu = dataBatch['img'][0,:]
            N = img_cpu.shape[0]
            imBatch = Variable(torch.FloatTensor(N, 1024, 14, 14))
            imBatch = imBatch.cuda(device)
            # if args.batch_size == 1:
            #     img_list = to_image_list(img_cpu[0,:,:], cfg.DATALOADER.SIZE_DIVISIBILITY)
            # else:
            #     img_list = to_image_list(img_cpu, cfg.DATALOADER.SIZE_DIVISIBILITY)
            # print(cfg.DATALOADER.SIZE_DIVISIBILITY)
            # img_list = to_image_list(img_cpu, cfg.DATALOADER.SIZE_DIVISIBILITY)
            # img_list = to_image_list(img_cpu)
            imBatch.data.copy_(img_cpu)  # Tensor.shape(BatchSize, 3, Height, Width)

            target_cpu = dataBatch['target']
            # print(target_cpu)
            targetBatch.data.copy_(target_cpu)
            #print(imBatch.shape)
            #print(targetBatch.shape)

            # Train networ
            optimizer.zero_grad()

            # pred = model(features_roi, features_backbone)
            pred = model(imBatch)

            # print('target:', targetBatch[0,:][0])
            loss = criterion(pred, targetBatch)
            action = pred.cpu().argmax(dim=1).data.numpy()

            loss.backward()

            optimizer.step()
            accuracy += np.sum(action==targetBatch.cpu().data.numpy())

            lossArr.append(loss.cpu().data.item())
            AccuracyArr.append(accuracy / iteration /args.batch_size)

            meanLoss = np.mean(np.array(lossArr))
            if iteration % 100 == 0:
                print('prediction:', pred)
                print('predicted action:', action)
                print('ground truth:', targetBatch.cpu().data.numpy())
                print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                epoch, iteration, lossArr[-1], meanLoss))

                trainingLog.write('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f \n' % (
                epoch, iteration, lossArr[-1], meanLoss))

                print('Epoch %d Iteration %d: Accumulated Accuracy %.5f' % (epoch, iteration, AccuracyArr[-1]))
                trainingLog.write(
                    'Epoch %d Iteration %d: Accumulated Accuracy %.5f \n' % (epoch, iteration, AccuracyArr[-1]))

            if epoch in [50, 70] and iteration == 1:
                print('The learning rate is being decreased at Iteration %d', iteration)
                trainingLog.write('The learning rate is being decreased at Iteration %d \n' % iteration)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

            if iteration == args.MaxIteration and epoch % 5 == 0:
                torch.save(model.state_dict(), (outdir + 'netFinal_%d.pth' % (epoch + 1)))
                break

        if iteration >= args.MaxIteration:
            break

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), (outdir + 'netFinal_%d.pth' % (epoch + 1)))
        if args.val and epoch % 10 == 0:
            print("validation")
            test(cfg, args)



def run_test():
    pass


def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    )
    parser.add_argument(
        "--is_cat",
        default=False,
        help="If we use concatenation on object features",
        type=bool,

    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--MaxIteration",
        help="the iteration to end training",
        type=int,
        default=90000,
    )
    parser.add_argument(
        "--initLR",
        help="Initial learning rate",
        default=0.001
    )
    parser.add_argument(
        "--imageroot",
        type=str,
        help="Directory to the images",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/features/train"
    )
    parser.add_argument(
        "--gtroot",
        type=str,
        help="Directory to the groundtruth",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/annotations/train_gt_action.json"
    )
    parser.add_argument(
        "--imWidth",
        type=int,
        help="Crop to width",
        default=1280
    )
    parser.add_argument(
        "--imHeight",
        type=int,
        help="Crop to height",
        default=720
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size",
        default=1
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Give this experiment a name",
        default=str(datetime.datetime.now())
    )
    parser.add_argument(
        "--model_root",
        type=str,
        help="Directory to the trained model",
        default="."
    )
    parser.add_argument(
        "--val",
        action='store_true',
        default=False,
        help='Validation or not'
    )

    args = parser.parse_args()
    print(args)
    cfg.MODEL.WEIGHT = "/data6/SRIP_SelfDriving/Outputs/model_final.pth"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if torch.cuda.is_available():
        print("CUDA device is available.")

    # output directory
    outdir = cfg.OUTPUT_DIR
    print("Save path:", outdir)
    if outdir:
        mkdir(outdir)

    #    logger = setup_logger("training", outdir)

    train(cfg, args)

    # if validate
    if not args.skip_test:
        run_test()


if __name__ == "__main__":
    main()
