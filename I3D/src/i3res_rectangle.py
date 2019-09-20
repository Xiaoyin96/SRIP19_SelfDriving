import math

import torch
from torch.nn import ReplicationPad3d
import torchvision
import copy
# from torchvision import datasets, transforms

from src import inflate



class I3ResNet(torch.nn.Module):
    def __init__(self, resnet2d, frame_nb=32, class_nb=4, reason_nb=21, side_task=False, conv_class=False):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.conv_class = conv_class

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=5, time_stride=1, time_padding=2, center=True)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=1, time_stride=1)
        self.maxpool2 = torch.nn.MaxPool3d(kernel_size=(3,1,1),stride=(2,1,1), padding=(1,0,0))

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)
        self.side_task = side_task

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True)

            if self.side_task:
                self.classifier2 = torch.nn.Conv3d(
                    in_channels=2048,
                    out_channels=reason_nb,
                    kernel_size=(1, 1, 1),
                    bias=True)
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = inflate.inflate_linear(resnet2d.fc, 1)
        

    def forward(self, x):
        x = self.conv1(x) # input size: batch_size x 3 x nf x 640 x 360
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.maxpool(x) # output: bs x 64 x nf x 160 x 90

        x = self.layer1(x) # output: bs x 256 x nf x 160 x 90
        # print('layer1:', x.shape)
        x = self.maxpool2(x) # output: bs x 256 x nf x 160 x 90
        # print('maxpool2:', x.shape)
        x = self.layer2(x) # output: bs x 512 x nf/2 x 80 x 45
        # print('layer2:', x.shape)
        x = self.layer3(x) # output: bs x 1024 x nf/2 x 40 x 23 --> option 1: extract feature map
        # print('layer3:', x.shape)
        x = self.layer4(x) # output: bs x 2048 x nf/2 x 20 x 12 
        # print('layer4:', x.shape)

        if self.conv_class:
            x = self.avgpool(x)
            r1 = self.classifier(x) # output: bs x 4 x 1 x 1 x 1
            r1 = r1.squeeze(3)
            r1 = r1.squeeze(3) # squeeze: bs x 4 x 1
            r1 = r1.mean(2) # bs x 4 (batch size x class_nb)
            if self.side_task:
                r2 = self.classifier2(x)
                r2 = r2.squeeze(3)
                r2 = r2.squeeze(3) 
                r2 = r2.mean(2)        
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            r1 = self.fc(x_reshape)
        return r1 # if not self.side_task, else: r1, r2
       

def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        # spatial_stride = bottleneck2d.conv2.stride[0]
        spatial_stride = 1

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=3, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=1,
            time_padding=1,
            time_stride=spatial_stride,
            center=True)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x  
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) 
        out = self.bn3(out)

        if self.downsample is not None: 
            residual = self.downsample(x) 

        out += residual # element wise addition
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d


# resnet2d = torchvision.models.resnet101(pretrained=True)
# i3d = I3ResNet(copy.deepcopy(resnet2d), 32, 4, conv_class=True)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# i3d = i3d.to(device)
# input = torch.ones([1,3,32,224,224], dtype=torch.float, device=device)
# output = i3d(input)
# print(i3d)