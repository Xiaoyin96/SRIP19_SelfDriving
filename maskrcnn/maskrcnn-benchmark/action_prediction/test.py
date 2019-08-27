import argparse
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list

from train import SimpleHook
from baseline import baseline
from dataloader import BatchLoader

def test(cfg, args):
    # load detector
    detector = build_detection_model(cfg)
    detector.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    detector.to(device)
    outdir = cfg.OUTPUT_DIR

    # load network
    model = baseline()
    model.load_state_dict(torch.load(args.model_root))

    # Initialize image batch
    # imBatch = Variable(torch.FloatTensor(args.batch_size, 3, args.imHeight, args.imWidth))
    imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 736, 1280))
    targetBatch = Variable(torch.LongTensor(args.batch_size, 1))

    # Move network and batch to gpu
    imBatch = imBatch.cuda(device)
    targetBatch = targetBatch.cuda(device)
    model = model.cuda(device)

    # Initialize dataloader
    Dataset = BatchLoader(
            imageRoot = args.imageroot,
            gtRoot = args.gtroot,
            cropSize = (args.imWidth, args.imHeight)
            )
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    length = Dataset.__len__()

    AccuracyArr = []
    accuracy = 0

    # test
    SaveFilename = (outdir + 'TestingLog.txt')
    TestingLog = open(SaveFilename, 'w')
    print('Save to ', SaveFilename)
    for i, dataBatch in enumerate(dataloader):
        # Read data, under construction. now it is hard-code
        img_cpu = dataBatch['img']
        img_list = to_image_list(img_cpu[0, :, :], cfg.DATALOADER.SIZE_DIVISIBILITY)
        imBatch.data.copy_(img_list.tensors)  # Tensor.shape(BatchSize, 3, Height, Width)

        target_cpu = dataBatch['target']
        # print(target_cpu)
        targetBatch.data.copy_(target_cpu)

        # grap features from detector
        RoIPool_module = detector.roi_heads.box.feature_extractor.pooler
        Backbone = detector.backbone
        hook_roi = SimpleHook(RoIPool_module)
        hook_backbone = SimpleHook(Backbone)
        out_detector = detector(imBatch)
        features_roi = hook_roi.output.data
        features_backbone = hook_backbone.output[0].data  # only use the bottom one

        pred = model(features_roi, features_backbone)
        action = pred.cpu().argmax().data.numpy()

        print('predicted action:', action)
        print('ground truth:', target_cpu.data.numpy()[0])

        if action == target_cpu.data.numpy()[0]:
            accuracy += 1

        AccuracyArr.append(accuracy/(i + 1))

        print('Iteration %d / %d: Accumulated Accuracy %.5f' % (i + 1, length, AccuracyArr[-1]))
        TestingLog.write('Iteration %d / %d: Accumulated Accuracy %.5f \n' % (i + 1, length, AccuracyArr[-1]))



def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_101_FPN_1x.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--imageroot",
        type=str,
        help="Directory to the images",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/images/100k/val"
    )
    parser.add_argument(
        "--gtroot",
        type=str,
        help="Directory to the groundtruth",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/annotations/val_gt_action.json"
    )
    parser.add_argument(
        "--model_root",
        type=str,
        help="Directory to the trained model",
        default="/data6/SRIP_SelfDriving/Outputs/baseline/netFinal_10.pth"
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

    args = parser.parse_args()
    print(args)

    cfg.OUTPUT_DIR = "/data6/SRIP_SelfDriving/Outputs/"
    cfg.MODEL.WEIGHT = "/data6/SRIP_SelfDriving/Outputs/model_final.pth"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    print(cfg)
    cfg.freeze()

    if torch.cuda.is_available():
        print("CUDA device is available.")

    test(cfg, args)

if __name__ == "__main__":
    main()
