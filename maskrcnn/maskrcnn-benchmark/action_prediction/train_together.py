import argparse
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from network_together import Network
from DataLoader_together import BatchLoader

def train(cfg, args):
    # torch.cuda.set_device(2)

    # Initialize the network
    print(args.is_cat)
    model = Network(cfg, is_cat = args.is_cat)
    print(model)
    
#    torch.distributed.init_process_group(backend="nccl", init_method="env://")
#    model = nn.parallel.DistributedDataParallel(model)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # TODO: how to input weights
    outdir = cfg.OUTPUT_DIR

    class_weights = [1, 1, 1, 1, 1]
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(args.initLR), weight_decay=float(args.weight_decay))

    # Initialize image batch
    imBatch = Variable(torch.FloatTensor(args.batch_size, 3, args.imHeight, args.imWidth))
    targetBatch = Variable(torch.FloatTensor(args.batch_size, 5))

    # Move network and batch to GPU
    imBatch = imBatch.cuda(device)
    targetBatch = targetBatch.cuda(device)
    model = model.cuda(device)

    # Initialize DataLoader
    Dataset = BatchLoader(
        imageRoot = args.imageroot,
        gtRoot = args.gtroot,
        cropSize = (args.imHeight, args.imWidth)

    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=0, shuffle=True)

    lossArr = []
    AccuracyArr = []


    for epoch in range(0, args.num_epoch):
        trainingLog = open(outdir + ('trainingLogTogether_{0}.txt'.format(epoch)), 'w')
        trainingLog.write(str(args))
        accuracy = 0
        for i, dataBatch in enumerate(dataloader):

            # Read data
            img_cpu = dataBatch['img']
            imBatch.data.copy_(img_cpu)

            target_cpu = dataBatch['target']
            targetBatch.data.copy_(target_cpu)

            optimizer.zero_grad()
            pred = model(imBatch)

            loss = criterion(pred, targetBatch)

            loss.backward()
            optimizer.step()

            lossArr.append(loss.cpu().data.item())
            meanLoss = np.mean(np.array(lossArr))

            # Calculate accuracy
            predict = torch.sigmoid(pred) > 0.5
            # y_pred = (predict == target_cpu)
            f1 = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='weighted')
            AccuracyArr.append(f1)
            if(len(AccuracyArr) > 100):
                AccuracyArr.pop(0)

            print('prediction logits:', pred)
            print('predicted action:', predict)
            print('ground truth:', targetBatch.cpu().data.numpy())
            print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                    epoch, i, lossArr[-1], meanLoss))
            print('Epoch %d Iteration %d: F1 %.5f Accumulated F1 %.5f' % (
                    epoch, i, AccuracyArr[-1], np.mean(np.array(AccuracyArr))))
            if i % 100 == 0:
                trainingLog.write('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f \n' % (
                    epoch, i, lossArr[-1], meanLoss))
                #print('Epoch %d Iteration %d: F1 %.5f Accumulated F1 %.5f' % (
                    #epoch, i, AccuracyArr[-1], np.mean(np.array(AccuracyArr))))

            if epoch in [int(0.5*args.num_epoch), int(0.7*args.num_epoch)] and i==0:
                print('The learning rate is being decreased at Iteration %d', i)
                trainingLog.write('The learning rate is being decreased at Iteration %d \n' % i)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

            if i == args.MaxIteration and epoch % 1 == 0:
                torch.save(model.state_dict(), (outdir + 'netFinal_%d.pth' % (epoch + 1)))
                break

        if i >= args.MaxIteration:
            break

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), (outdir + 'netFinal_%d.pth' % (epoch + 1)))
        if args.val and epoch % 10 == 0:
            print("Validation...")
            run_test(cfg, args)


def run_test(cfg, args):
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
        default=True
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
        default=0.001,
        type=float
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
    parser.add_argument(
        "--num_epoch",
        default=20,
        help="The number of epoch for training",
        type=int
    )
    
#    parser.add_argument('--rank', type=int, default=0)
#    parser.add_argument('--world-size', type=int, default=1)
#    parser.add_argument('--local_rank', type=int)
    
    args = parser.parse_args()
    print(args)
    # cfg.MODEL.WEIGHT = "/data6/SRIP_SelfDriving/Outputs/model_final.pth"
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
