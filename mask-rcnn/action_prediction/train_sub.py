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
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list

from baseline_sub import baseline
from dataloader_sub import BatchLoader

import torch.optim as optim


def train(cfg, args):
    detector = build_detection_model(cfg)
    print(detector.roi_heads.head)
    detector.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    detector.to(device)
    outdir = cfg.OUTPUT_DIR

    # Initialize the network
    model = baseline()
    class_weights = [1, 1, 5, 5]  # could be adjusted
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize optimizer
    #optimizer = optim.SGD(model.parameters(), lr=args.initLR, momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=float(args.initLR), weight_decay=0.001)

    # Initialize image batch
    # imBatch = Variable(torch.FloatTensor(args.batch_size, 3, args.imHeight, args.imWidth))
    imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 736, 1280))
    targetBatch = Variable(torch.LongTensor(args.batch_size))

    # Move network and batch to gpu
    imBatch = imBatch.cuda(device)
    targetBatch = targetBatch.cuda(device)
    model = model.cuda(device)

    # Initialize dataloader
    Dataset = BatchLoader(
        imageRoot=args.imageroot,
        gtRoot=args.gtroot,
        #cropSize=(args.imWidth, args.imHeight)
    )
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    lossArr = []
    AccuracyArr = []
    accuracy = 0
    iteration = 0

    for epoch in range(0, 10):
        trainingLog = open(outdir + ('trainingLog_{0}.txt'.format(epoch)), 'w')
        accuracy = 0
        for i, dataBatch in enumerate(dataloader):
            iteration = i + 1

            # Read data, under construction
            img_cpu = dataBatch['img']
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

            # Train networ
            RoIPool_module = detector.roi_heads.box.feature_extractor.pooler
            RoIPredictor = detector.roi_heads.box.predictor
            Backbone = detector.backbone
            RoIProc = detector.roi_heads.box.post_processor
            hook_roi = SimpleHook(RoIPool_module)
            hook_backbone = SimpleHook(Backbone)
            hook_pred = SimpleHook(RoIPredictor)
            hook_proc = SimpleHook(RoIProc)
            out_detector = detector(imBatch)

            features_roi = hook_roi.output.data
            features_backbone = hook_backbone.output[0].data  # only use the bottom one
            boxes = hook_proc.input[1] # boxes BoxList
            features_roi_list = split_boxes(boxes, features_roi)



            # thresh = 10
            # cls_logit = hook_pred.output[0].data
            # cls_logit = torch.max(cls_logit, dim=1)
            # # print(cls_logit)
            # ind = torch.ge(cls_logit[0], torch.FloatTensor([thresh]).to(device))
            # # print(ind)
            # features_roi = features_roi[ind]
            # print(features_roi.shape)
            optimizer.zero_grad()

            # pred = model(features_roi, features_backbone)
            pred = model(features_roi_list, features_backbone)

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

            if epoch in [4, 7] and iteration == 1:
                print('The learning rate is being decreased at Iteration %d', iteration)
                trainingLog.write('The learning rate is being decreased at Iteration %d \n' % iteration)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

            if iteration == args.MaxIteration:
                torch.save(model.state_dict(), (outdir + 'netFinal_%d.pth' % (epoch + 1)))
                break

        if iteration >= args.MaxIteration:
            break

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), (outdir + 'netFinal_%d.pth' % (epoch + 1)))

def split_boxes(boxes, features):
    length1 = 0
    features_roi_list = []
    for box in boxes:
        length2 = len(box) + length1
        feature_roi = features[length1:length2, :]
        features_roi_list.append(feature_roi)
        length1 = length2

    return features_roi_list




class SimpleHook(object):
    """
    A simple hook function to extract features.
    :return:
    """

    def __init__(self, module, backward=False, other=False):
        # super(SimpleHook, self).__init__()
        self.other = other
        if not backward:
            if not other:
                self.hook = module.register_forward_hook(self.hook_fn)
            else:
                self.hook = module.register_backward_hook(self.hook_fn_other)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def hook_fn_other(self, module, input, output):
        self.box_per_img = module.box_per_img

    def close(self):
        self.hook.remove()


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
        default=0.0001
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
        default=2
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Give this experiment a name",
        default=str(datetime.datetime.now())
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
