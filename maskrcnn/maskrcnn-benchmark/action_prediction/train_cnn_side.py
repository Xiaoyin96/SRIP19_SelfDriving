import argparse
import os
import datetime
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import f1_score

from maskrcnn_benchmark.utils.miscellaneous import mkdir
from dataloader_cnn_side import BatchLoader

class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        res101 = models.resnet101(pretrained=True)
        num_ftrs = res101.fc.in_features
        modules = list(res101.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(num_ftrs, 5)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(num_ftrs, 21)
        self.drop2 = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        r1 = self.drop1(self.fc1(x))
        r2 = self.drop2(self.fc2(x))
        #r2 = self.fc2(x)
        return r1, r2

def train(args):
    # torch.cuda.set_device(3)


    # Initialize the network
    model = baseline()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.model_root))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)   
    
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outdir = args.out_dir
    
    class_weights = [0.4, 2, 2, 2, 1]
    w = torch.FloatTensor(class_weights).cuda()
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion2 = nn.BCEWithLogitsLoss().cuda()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(args.initLR), weight_decay=float(args.weight_decay))

    # Initialize image batch
#    imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 224, 224))
#    targetBatch = Variable(torch.FloatTensor(args.batch_size, 5))

    # Move network and batch to GPU
#    imBatch = imBatch.to(device)
#    targetBatch = targetBatch.to(device)
    #model = model.cuda(device)

    # Initialize DataLoader
    Dataset = BatchLoader(
        imageRoot = args.imageroot,
        gtRoot = args.gtroot,
        reasonRoot = args.reasonroot,
        #cropSize = (args.imHeight, args.imWidth)

    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=4, shuffle=True)

    lossArr = []
    AccuracyArr = []


    for epoch in range(0, args.num_epoch):
        lossArr = []
        AccuracyArr = []

        for i, dataBatch in enumerate(dataloader):
            # Read data
            img_cpu = dataBatch['img']
            imBatch = img_cpu.to(device)
            # imBatch.data.copy_(img_cpu)

            target_cpu = (dataBatch['target']).type(torch.FloatTensor)
            targetBatch = target_cpu.to(device)
            # targetBatch.data.copy_(target_cpu)
            reason_cpu = (dataBatch['reason']).type(torch.FloatTensor)
            reasonBatch = reason_cpu.to(device)

            optimizer.zero_grad()
            pred, pred_reason = model(imBatch)

            loss1 = criterion(pred, targetBatch)
            loss2 = criterion2(pred_reason, reasonBatch)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            loss_cpu = np.array(loss.cpu().data.item())

            lossArr.append(loss_cpu)
            meanLoss = np.mean(np.array(lossArr))

            # Calculate accuracy
            predict = torch.sigmoid(pred) > 0.5
            f1 = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
            AccuracyArr.append(f1)
            predict_reason = torch.sigmoid(pred_reason) > 0.5
            f1_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')

            if i % 50 == 0:
                #print('prediction:', pred)
                print('prediction logits:{}'.format(predict))

                print('ground truth:{}'.format(targetBatch.cpu().data.numpy()))
                print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                    epoch, i, lossArr[-1], meanLoss))
                print('Epoch %d Iteration %d: F1 %.5f Accumulated F1 %.5f' % (
                    epoch, i, AccuracyArr[-1], np.mean(np.array(AccuracyArr))))
                print('Reason F1 score:{}'.format(f1_reason))

            if epoch in [int(0.5*args.num_epoch), int(0.7*args.num_epoch)] and i==0:
                print('The learning rate is being decreased at Iteration %d', i)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

        if i >= args.MaxIteration:
            break

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), (outdir + 'net_%d.pth' % (epoch + 1)))
        if args.val and (epoch + 1)% 5 == 0:
            print("Validation...")
            run_test(args, model, device)

    torch.save(model.state_dict(), (outdir + 'net_Final.pth'))

def run_test(args, model, device):
    # Initialize image batch
    imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 224, 224))
    targetBatch = Variable(torch.FloatTensor(args.batch_size, 5))

    # Move network and batch to GPU
    imBatch = imBatch.to(device)
    targetBatch = targetBatch.to(device)

    Dataset = BatchLoader(
        imageRoot = args.imageroot,
        gtRoot = args.val_gtroot,
        #cropSize = (args.imHeight, args.imWidth)

    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=4, shuffle=True)

    model.eval()

    AccuracyArr = []

    for i, dataBatch in enumerate(dataloader):
        # tic = time.time()
        # Read data
        img_cpu = dataBatch['img']
        imBatch.data.copy_(img_cpu)

        target_cpu = dataBatch['target']
        targetBatch.data.copy_(target_cpu)

        pred = model(imBatch)

        # Calculate accuracy
        predict = torch.sigmoid(pred) > 0.5

        f1 = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
        AccuracyArr.append(f1)


    print("Validation F1 score:", AccuracyArr[-1])
    model.train()


def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/configs/baseline.yaml",
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
        default=0.003,
        help="Weight decay",
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
        default=0.00005
    )
    parser.add_argument(
        "--imageroot",
        type=str,
        help="Directory to the images",
        default="/data6/SRIP19_SelfDriving/bdd12k/data1/"
    )
    parser.add_argument(
        "--gtroot",
        type=str,
        help="Directory to the groundtruth",
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/12k_gt_train_5_actions.json"
    )
    parser.add_argument(
        "--reasonroot",
        type=str,
        help="Directory to the explanations",
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/train_reason_img.json"
    )
    parser.add_argument(
        "--val_gtroot",
        type=str,
        help="Directory to the groundtruth",
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/12k_gt_val_5_actions.json"
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
        default=8
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
        "--checkpoint",
        type=str,
        help="If using checkpoint weights",
        default=False
    
    )
    parser.add_argument(
        "--val",
        action='store_true',
        default=False,
        help='Validation or not'
    )
    parser.add_argument(
        "--num_epoch",
        default=50,
        help="The number of epoch for training",
        type=int
    )
    parser.add_argument(
        "--out_dir",
        default='/data6/SRIP19_SelfDriving/bdd12k/CNN/',
        help="output directory",
        type=str
    )

    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        print("CUDA device is available.")

    # output directory
    outdir = args.out_dir
    print("Save path:", outdir)
    if outdir:
        mkdir(outdir)

    #    logger = setup_logger("training", outdir)

    train(args)

if __name__ == "__main__":
    main()
