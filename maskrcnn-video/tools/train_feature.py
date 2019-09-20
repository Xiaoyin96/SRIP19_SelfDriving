import argparse
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import f1_score

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model

from feature_loader import FeatureLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def train(cfg, args):
    # torch.cuda.set_device(2)

    # Initialize the network
    model = build_detection_model(cfg)
    bbox_dict = torch.load(args.bboxRoot) # last frames's bbox gt
    bbox_dict = bbox_dict['train']
    print('loaded bbox ground truth...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outdir = cfg.OUTPUT_DIR

    class_weights = [1, 2, 2, 2]
    w = torch.FloatTensor(class_weights).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda() # pos_weight=w

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(args.initLR), weight_decay=float(args.weight_decay))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize DataLoader
    Dataset = FeatureLoader(
        featureRoot=args.featureRoot,
        actionRoot=args.actionRoot
    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=0, shuffle=True)

    # training
    for epoch in range(0, args.num_epoch):

        lossArr = []
        AccuracyArr = []
        for i, dataBatch in enumerate(dataloader):

            # Read data
            vid =  dataBatch['vid'][0]
            feature_cpu = dataBatch['feature'] # 16 x 1024 x 14 x 14
            action_cpu = dataBatch['action'] 
            bbox_cpu =  bbox_dict[vid] #  one BoxList
            if not bbox_cpu or len(feature_cpu.size())==0:
                continue

            feature = feature_cpu.to(device)
            bbox = bbox_cpu.to(device)
            action = action_cpu.to(device)
            action = torch.squeeze(action) # torch.size([4])

            optimizer.zero_grad()
            
            pred = model(feature, bbox, bbox) # torch.size([4])
            loss = criterion(pred, action)

            if loss >= 10:
                print('bad data:',vid)

            loss.backward()
            optimizer.step()
            loss_cpu = loss.cpu().data.item()

            lossArr.append(loss_cpu)
            meanLoss = np.mean(np.array(lossArr))

            # Calculate accuracy
            predict = torch.sigmoid(pred) >= 0.5

            f1 = f1_score(action.cpu().data.numpy(), predict.cpu().data.numpy(), average='macro')
            AccuracyArr.append(f1)
            meanAcc = np.mean(np.array(AccuracyArr))

            if i % 10 == 0:
                print('prediction logits:', predict)
                print('ground truth:', action.cpu().data.numpy())
                print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                    epoch, i, loss_cpu, meanLoss))
                print('Epoch %d Iteration %d Action Prediction: F1 %.5f Accumulated F1 %.5f' % (
                    epoch, i, AccuracyArr[-1], meanAcc))
            
            if epoch in [int(0.3 * args.num_epoch),int(0.5 * args.num_epoch), int(0.7 * args.num_epoch)] and i == 0:
                print('The learning rate is being decreased at Iteration %d' % i)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                
        # scheduler.step() # decrease lr to 0.1lr every 5 epoches

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), (outdir + 'net_%d.pth' % (epoch + 1)))
        
    print("Saving final model...")
    torch.save(model.state_dict(), (outdir + 'net_Final.pth'))
    print("Done!")

        

def run_test(cfg, args):
    pass

# def DrawBbox(img, boxlist):
#     plt.imshow(img)
#     currentAxis = plt.gca()
#     for i in range(boxlist.shape[0]):
#         bbox = boxlist[i]
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
#         currentAxis.add_patch(rect)

#     plt.show()

def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="/home/selfdriving/maskrcnn-benchmark/configs/e2e_faster_rcnn_I3D_resnet101.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0001,
        help="Weight decay",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--initLR",
        help="Initial learning rate",
        default=0.001
    )
    parser.add_argument(
        "--freeze",
        default=False,
        help="If freeze faster rcnn",
    )
    parser.add_argument(
        "--featureRoot",
        type=str,
        help="Directory to the features",
        default="/home/selfdriving/I3D/features/"
    )
    parser.add_argument(
        "--bboxRoot",
        type=str,
        help="Directory to the bbox groundtruth",
        default="/home/selfdriving/mrcnn/output/inference/bdd100k_val/last_preds.pth"
    )
    parser.add_argument(
        "--actionRoot",
        type=str,
        help="Directory to the action label groundtruth",
        default="/home/selfdriving/I3D/data/4action_reason.json"
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
        default="/home/selfdriving/mrcnn/model_final_apt.pth"
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
    parser.add_argument(
        "--from_checkpoint",
        default=False,
        help="If we need load weights from checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=".",
        help="The path to the checkpoint weights.",
        type=str,
    )

    args = parser.parse_args()
    print(args)
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


if __name__ == "__main__":
    main()
