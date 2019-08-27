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

from utils import attention
import matplotlib.pyplot as plt

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
        #self.drop2 = nn.Dropout(0.25)
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        r1 = self.drop1(self.fc1(x))
        r2 = self.fc2(x)
        return r1, r2


def test(args):
    # Initialize the network
    model = baseline()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_root))
    
    print(model)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outdir = args.out_dir
    
    if args.is_savemaps:
        hook_conv5 = SimpleHook(model.layer4)
    
    # Initialize image batch
    #imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 224, 224))
    #targetBatch = Variable(torch.FloatTensor(args.batch_size, 5))
    
    # Move network and batch to GPU
    #imBatch = imBatch.to(device)
    #targetBatch = targetBatch.to(device)
    
    # Initialize DataLoader
    Dataset = BatchLoader(
        imageRoot = args.imageroot,
        gtRoot = args.gtroot,
        reasonRoot = args.reasonroot,
        # cropSize = (args.imHeight, args.imWidth)

    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=8, shuffle=False)
    
    
    AccuracyArr = []
    AccOverallArr = []
    RandomAcc = []
    ReasonAcc = []
    
    SaveFilename = (outdir + 'TestingLog.txt')
    TestingLog = open(SaveFilename, 'w')
    print('Save to ', SaveFilename)
    TestingLog.write(str(args) + '\n')
    
    for i, dataBatch in enumerate(dataloader):
        # Read data
        img_cpu = dataBatch['img']
        imBatch = img_cpu.to(device)
        ori_img_cpu = dataBatch['ori_img']

        target_cpu = dataBatch['target']
        targetBatch = target_cpu.to(device)
        reason_cpu = dataBatch['reason']
        reasonBatch = reason_cpu.to(device)
 
        pred, pred_reason = model(imBatch)
        
        if args.is_savemaps:
            hooked_features = hook_conv5.output.data
            hooked_features = torch.mean(torch.mean(hooked_features, dim=0 ), dim=0)
            # print(hooked_features.shape)
            new_img = attention(ori_img_cpu.squeeze(0).data.numpy(), hooked_features.cpu().data.numpy())
            plt.imsave((outdir + 'att_maps/' + str(i) + '.jpg'), new_img)
        
        # Calculate accuracy
        predict = torch.sigmoid(pred) > 0.5
        predict_reason = torch.sigmoid(pred_reason) > 0.5

        f1 = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average=None)
        f1_overall = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
        f1_reason = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')

        # print("f1 score:{}".format(f1))
        AccuracyArr.append(f1)
        AccOverallArr.append(f1_overall)
        # print(AccuracyArr)
        ReasonAcc.append(f1_reason)

        # random guess
        random = torch.randint(0,2,(args.batch_size,5))
        random[:,4] = 0
        random_f1 = f1_score(target_cpu.data.numpy(), random.cpu().data.numpy(), average=None)
        RandomAcc.append(random_f1)

        print('prediction logits:', pred)
        print('prediction action: \n {}'.format(predict))
        print('ground truth: \n', targetBatch.cpu().data.numpy())
        print('Iteration {}: F1 {} Accumulated F1 {}' .format(
        i, AccuracyArr[-1], np.mean(np.array(AccuracyArr),axis=0)))

        
        TestingLog.write('prediction logits:' + str(pred) + '\n')
        TestingLog.write('prediction action: \n {}'.format(predict) + '\n')
        TestingLog.write('ground truth: \n' + str(targetBatch.cpu().data.numpy()) + '\n')
        TestingLog.write('Iteration {}: F1 {} Accumulated F1 {}' .format(
                          i, AccuracyArr[-1], np.mean(np.array(AccuracyArr),axis=0)) + '\n')
        TestingLog.write('\n')
    
    
    print("Random guess acc:{}".format(np.mean(np.array(RandomAcc),axis=0)))
    print("Overall acc:{}".format(np.mean(np.array(AccOverallArr),axis=0)))
    print("Reason acc:{}".format(np.mean(np.array(ReasonAcc),axis=0)))
    TestingLog.write("Random guess acc:{}".format(np.mean(np.array(RandomAcc),axis=0)) + '\n')
    TestingLog.write("Overall acc:{}".format(np.mean(np.array(AccOverallArr),axis=0)) + '\n')
    
    TestingLog.close()

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
    parser = argparse.ArgumentParser(description="Action Prediction testing")
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
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/12k_gt_test_5_actions.json"
    )
    parser.add_argument(
        "--reasonroot",
        type=str,
        help="Directory to the groundtruth",
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/test_reason_img.json"
    )
    parser.add_argument(
        "--model_root",
        type=str,
        help="Directory to the trained model",
        default='/data6/SRIP19_SelfDriving/bdd12k/BS8_2/net_5.pth',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size",
        default=1
    )
    parser.add_argument(
        "--is_savemaps",
        type=bool,
        help="Whether save attention maps",
        default=False
    )
    parser.add_argument(
        "--out_dir",
        default='/data6/SRIP19_SelfDriving/bdd12k/CNN/test/',
        help="output directory",
        type=str
    )
    args = parser.parse_args()
    print(args)
    
    test(args)
    
if __name__ == "__main__":
    main()
