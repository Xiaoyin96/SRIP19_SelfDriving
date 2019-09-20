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
from maskrcnn_benchmark.modeling.detector import build_detection_model

from feature_loader import FeatureLoader

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

def test(cfg, args):
    # Initialize the network
    model = build_detection_model(cfg)
    state_dict = torch.load(args.model_root)
    model.load_state_dict(state_dict) # load pretrained model

    bbox_dict = torch.load(args.bboxRoot) # last frames's bbox gt
    bbox_dict = bbox_dict['val']
    print('loaded test set bbox ground truth...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outdir = cfg.OUTPUT_DIR

    # Initialize test DataLoader
    Dataset = FeatureLoader(
        featureRoot=args.featureRoot,
        actionRoot=args.actionRoot
    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=24, shuffle=True)

    # Forward test set
    model.eval()
    lossArr = []
    AccuracyArr = []
    classAcc = np.zeros((1,4))
    with torch.no_grad():
        for i, dataBatch in enumerate(dataloader):

            # Read data
            vid =  dataBatch['vid'][0]
            feature_cpu = dataBatch['feature'] # last frame's feature
            action_cpu = dataBatch['action'] 
            bbox_cpu =  bbox_dict[vid] #  one BoxList
            if not bbox_cpu or len(feature_cpu.size())==0:
                continue

            feature = feature_cpu.to(device)
            bbox = bbox_cpu.to(device)
            action = action_cpu.to(device)
            action = torch.squeeze(action) # torch.size([4])

            pred = model(feature, bbox, bbox) # torch.size([4])

            # Calculate accuracy
            predict = torch.sigmoid(pred) >= 0.5
            action_np = action_cpu.data.numpy()
            predict_np = predict.cpu().data.numpy().reshape(1,4)

            f1 = f1_score(action_np, predict_np, average='macro')
            f1_class = f1_score(action_np, predict_np, average=None)
            
            AccuracyArr.append(f1)
            classAcc = np.vstack((classAcc,f1_class))
            meanAcc = np.mean(np.array(AccuracyArr))
            meanclsAcc = np.mean(classAcc, axis=0) 

            if i % 10 == 0:
                print('prediction logits:', predict_np)
                print('ground truth:', action_np)
                print('Iteration %d Action Prediction: F1 %.5f Accumulated F1 %.5f' % (
                    i, AccuracyArr[-1], meanAcc))
                print('Iteration {} Action Prediction:  Accumulated F1 class{}'.format(
                    i, meanclsAcc))



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
        default=1e-4,
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
        default=0.0001
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
        default="/home/selfdriving/I3D/features_val/"
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
        default="/home/selfdriving/mrcnn/video_output/model3/net_Final.pth"
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

    test(cfg, args)


if __name__ == "__main__":
    main()