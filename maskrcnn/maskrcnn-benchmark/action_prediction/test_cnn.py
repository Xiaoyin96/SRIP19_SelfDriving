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
from dataloader_cnn import BatchLoader

from utils import attention
import matplotlib.pyplot as plt

def test(args):
    # Initialize the network
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 5)
    model.fc = nn.Sequential(
                 nn.Linear(num_ftrs, 4),
                 nn.Dropout(0.25))

    
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
    # imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 224, 224))
    # targetBatch = Variable(torch.FloatTensor(args.batch_size, 4))
    #
    # # Move network and batch to GPU
    # imBatch = imBatch.to(device)
    # targetBatch = targetBatch.to(device)
    
    # Initialize DataLoader
    Dataset = BatchLoader(
        imageRoot = args.imageroot,
        gtRoot = args.gtroot,
        # cropSize = (args.imHeight, args.imWidth)

    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=0, shuffle=True)
    
    
    # AccuracyArr = []
    AccOverallArr = []
    # RandomAcc = []
    TargetArr = []
    PredArr = []
    RandomArr = []

    SaveFilename = (outdir + 'TestingLog.txt')
    TestingLog = open(SaveFilename, 'w')
    print('Save to ', SaveFilename)
    TestingLog.write(str(args) + '\n')


    # count = dataloader.__len__()
    for i, dataBatch in enumerate(dataloader):
        print(i)
        # print((str(i + 1) + '/' + str(count)))
        # Read data
        img_cpu = dataBatch['img']
        imBatch = img_cpu.to(device)
        # imBatch.data.copy_(img_cpu)
        # ori_img_cpu = dataBatch['ori_img']

        target_cpu = dataBatch['target']
        targetBatch = target_cpu.to(device)
        # targetBatch.data.copy_(target_cpu)
        
        pred = model(imBatch)
        
        if args.is_savemaps:
            hooked_features = hook_conv5.output.data
            hooked_features = torch.mean(torch.mean(hooked_features, dim=0 ), dim=0)
            # print(hooked_features.shape)
            # new_img = attention(ori_img_cpu.squeeze(0).data.numpy(), hooked_features.cpu().data.numpy())
            plt.imsave((outdir + 'att_maps/' + str(i) + '.jpg'), new_img)
        
        # Calculate accuracy
        predict = torch.sigmoid(pred) > 0.5
        # print(target_cpu.data.numpy().shape)
        # print(predict.cpu().data.numpy().shape)
        TargetArr.append(target_cpu.data.numpy())
        PredArr.append(predict.cpu().data.numpy())


        # print(target_cpu.data.numpy().shape, predict.cpu().data.numpy().shape)
        # f1 = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average=None)
        # print('f1 done!')
        f1_overall = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')

        # print("f1 score:{}".format(f1))
        # AccuracyArr.append(f1)
        AccOverallArr.append(f1_overall)
        # print(AccuracyArr)

        # random guess
        # random = torch.randint(0,2,(args.batch_size,4))
        random = np.random.randint(0,2,(predict.shape[0],4))
        RandomArr.append(random)
        # random[:,4] = 0
        # random_f1 = f1_score(target_cpu.data.numpy(), random.cpu().data.numpy(), average=None)
        # RandomAcc.append(random_f1)

        print('prediction logits:', pred)
        print('prediction action: \n {}'.format(predict))
        print('ground truth: \n', targetBatch.cpu().data.numpy())
        #print('Iteration {}: F1 {} Accumulated F1 {}' .format(
        #i, AccuracyArr[-1], np.mean(np.array(AccuracyArr),axis=0)))

        
        TestingLog.write('prediction logits:' + str(pred) + '\n')
        TestingLog.write('prediction action: \n {}'.format(predict) + '\n')
        TestingLog.write('ground truth: \n' + str(targetBatch.cpu().data.numpy()) + '\n')
        #TestingLog.write('Iteration {}: F1 {} Accumulated F1 {}' .format(
        #                  i, AccuracyArr[-1], np.mean(np.array(AccuracyArr),axis=0)) + '\n')
        TestingLog.write('\n')
        # if i >= 978:
        #     break


    TargetArr = List2Arr(TargetArr)
    PredArr = List2Arr(PredArr)
    RandomArr = List2Arr(RandomArr)

    f1_pred = f1_score(TargetArr, PredArr, average=None)
    f1_rand = f1_score(TargetArr, RandomArr, average=None)


    # print("Random guess acc:{}".format(np.mean(np.array(RandomAcc),axis=0)))
    print("Random guess acc:{}".format(f1_rand))
    print("Category Acc:{}".format(f1_pred))
    print("Average Acc:{}".format(np.mean(f1_pred)))
    print("Overall acc:{}".format(np.mean(np.array(AccOverallArr),axis=0)))
    # TestingLog.write("Random guess acc:{}".format(np.mean(np.array(RandomAcc),axis=0)) + '\n')
    TestingLog.write("Overall acc:{}".format(np.mean(np.array(AccOverallArr),axis=0)) + '\n')
    
    TestingLog.close()

def List2Arr(List):
    Arr1 = np.array(List[:-1]).reshape(-1, 4)
    Arr2 = np.array(List[-1]).reshape(-1, 4)

    return np.vstack((Arr1, Arr2))

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
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/12k_gt_val_5_actions.json"
    )
    parser.add_argument(
        "--model_root",
        type=str,
        help="Directory to the trained model",
        default='/data6/SRIP19_SelfDriving/bdd12k/Outputs/resnet_trained/net_10.pth',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size",
        default=4
    )
    parser.add_argument(
        "--is_savemaps",
        type=bool,
        help="Whether save attention maps",
        default=False
    )
    parser.add_argument(
        "--out_dir",
        default='/data6/SRIP19_SelfDriving/bdd12k/Outputs/resnet_trained/inference/',
        help="output directory",
        type=str
    )
    args = parser.parse_args()
    print(args)
    
    test(args)
    
if __name__ == "__main__":
    main()
