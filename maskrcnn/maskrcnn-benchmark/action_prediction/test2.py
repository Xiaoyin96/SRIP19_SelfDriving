import argparse
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list

from baseline_sub import baseline
from dataLoader3 import BatchLoader

def test(cfg, args):
    # load detector
    #device = torch.device(cfg.MODEL.DEVICE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = cfg.OUTPUT_DIR

    # load network
    model = baseline(cfg)
    model.load_state_dict(torch.load(args.model_root))
    model.eval()

    # Initialize image batch
    # imBatch = Variable(torch.FloatTensor(args.batch_size, 3, args.imHeight, args.imWidth))
#    imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 736, 1280))
    targetBatch = Variable(torch.LongTensor(args.batch_size))

    # Move network and batch to gpu
#    imBatch = imBatch.cuda(device)
    #targetBatch = targetBatch.cuda(device)
    model = model.to(device)

    # Initialize dataloader
    Dataset = BatchLoader(
            imageRoot = args.imageroot,
            gtRoot = args.gtroot,
            cropSize = (args.imWidth, args.imHeight)
            )
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    length = Dataset.__len__()

    AccuracyArr = []
    accuracy = 0

    # test
    SaveFilename = (outdir + 'TestingLog.txt')
    TestingLog = open(SaveFilename, 'w')
    print('Save to ', SaveFilename)
    TestingLog.write(str(args))
    for i, dataBatch in enumerate(dataloader):
        # Read data, under construction. now it is hard-code
        img_cpu = dataBatch['img'][0]
        # N = img_cpu.shape[0]
        # imBatch = Variable(torch.FloatTensor(N, 1024, 14, 14))
        # imBatch = imBatch.cuda(device)
        # imBatch.data.copy_(img_cpu)  # Tensor.shape(BatchSize, 3, Height, Width)
        imBatch = img_cpu.to(device)


        target_cpu = dataBatch['target']
        # print(target_cpu)
        #targetBatch.data.copy_(target_cpu)
        targetBatch = target_cpu.to(device)

        # grap features from detector

        pred = model(imBatch)
        action = pred.cpu().argmax(dim=1).data.numpy()

        print('predicted action:', action)
        print('ground truth:', target_cpu.data.numpy())
        TestingLog.write('predicted action:' + str(action) + '\n')
        TestingLog.write('ground truth:' + str(target_cpu.data.numpy()) + '\n')
        accuracy = np.sum(action==targetBatch.cpu().data.numpy())
        AccuracyArr.append(accuracy / args.batch_size)
        meanAcc = np.mean(np.array(AccuracyArr))

        print('Iteration %d / %d: Accumulated Accuracy %.5f' % (i + 1, length, meanAcc))
        TestingLog.write('Iteration %d / %d: Accumulated Accuracy %.5f \n' % (i + 1, length, meanAcc))

class SimpleHook(object):
    """
    A simple hook function to extract features.
    :return:
    """
    def __init__(self, module, backward=False):
        # super(SimpleHook, self).__init__()
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input_, output_):
        self.input = input_
        self.output = output_

    def close(self):
        self.hook.remove()

def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/configs/baseline.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    ) # Just in order to get outdir
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
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/features/val"
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
        default="/data6/SRIP19_SelfDriving/Outputs/trained/netFinal_200.pth"
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

#    cfg.OUTPUT_DIR = "/data6/SRIP_SelfDriving/Outputs/"
#    cfg.MODEL.WEIGHT = "/data6/SRIP_SelfDriving/Outputs/model_final.pth"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
#    print(cfg)
    cfg.freeze()

    if torch.cuda.is_available():
        print("CUDA device is available.")

    test(cfg, args)

if __name__ == "__main__":
    main()
