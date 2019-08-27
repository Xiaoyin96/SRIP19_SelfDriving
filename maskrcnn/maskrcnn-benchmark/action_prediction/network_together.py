import torch
import torch.nn as nn

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

class Network(nn.Module):

    def __init__(self, config, select_num=10, class_num=5, is_cat=True):
        super(Network, self).__init__()
        self.detector = build_detection_model(config)
        self.detector.eval()
        self.detector.load_state_dict(torch.load(config.MODEL.WEIGHT)['model'], strict=False)
        # output_dir = config.OUTPUT_DIR
        # checkpointer = DetectronCheckpointer(config, self.detector, save_dir=output_dir)
        # ckpt = config.MODEL.WEIGHT
        # _ = checkpointer.load(ckpt, use_latest=True)
        
        for count, child in enumerate(self.detector.children()):
            count += 1
            print(count)
            print(child)
            if count < 4:
                for param in child.parameters():
                    param.requires_grad = False
        self.selector = Selector()
        self.select_num = select_num
        self.sftmax = nn.Softmax(dim=0)
        self.is_cat = is_cat

        # ResNet Head
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )


        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=14)
        # self.conv_glob = nn.Conv2d(1024, 2048, 2, stride=2)
        # self.relu_glob = nn.PReLU(2048)
        if self.is_cat:
            self.conv1 = nn.Conv2d(2048, 512, 1)
            # self.relu1 = nn.PReLU(512)
            self.relu1 = nn.ReLU(inplace=True)
            self.fc1 = nn.Linear(512*(self.select_num+1), 512) # select top 5
            # self.fc1_relu = nn.PReLU(512)
            self.fc1_relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(512, class_num)
        else:
            self.conv1 = nn.Conv2d(4096, 512, 1, 1)
            # self.relu1 = nn.PReLU(512)
            self.relu1 = nn.ReLU(inplace=True)
            self.fc1 = nn.Linear(512, 512)
            # self.fc1_relu = nn.PReLU(512)
            self.fc1_relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(512, class_num)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout()



    def forward(self, x):
        detect_results = self.detector(x) #TODO: decide detector results


        scores = self.selector(detect_results['roi_features'])
        scores, idx = torch.sort(scores, dim=0, descending=True)
        scores_logits = self.sftmax(scores)
        idx = idx[:self.select_num].reshape(self.select_num)
        select_features = detect_results['roi_features'][idx]
        select_features *= scores_logits[:self.select_num] # shape(select_num, 1024, 14, 14)
        
        del scores, idx, scores_logits
        
        select_features = self.head(select_features) # shape(select_num, 2048, 7, 7)
        # select_features = self.relu_glob(self.conv_glob(select_features))

        glob_feature = self.avgpool_glob(detect_results['glob_feature']) # shape(1, 1024, 14, 14)
        del detect_results
        
        # glob_feature = self.relu_glob(self.conv_glob(glob_feature)) # shape(1, 2048, 7, 7)
        glob_feature = self.head(glob_feature) # shape(1, 2048, 7, 7)
  
        if self.is_cat:
            x = torch.cat((select_features, glob_feature), dim=0) # shape(select_num+1, 2048, 7, 7)
            del select_features, glob_feature
            x = self.relu1(self.conv1(x)) # shape(select_num+1, 512, 5, 5) dim: 2048 -> 512
            x = self.avgpool(x) # shape(select_num+1, 512, 1, 1)
            x = x.reshape(1, -1) # shape(1, (select_num+1)*512)
            x = self.fc1_relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

        else:
            select_features = torch.sum(select_features, dim=0).unsqueeze(0) # shape(1, 2048, 7, 7)
            x = torch.cat((select_features, glob_feature), dim=1) # shape(1, 4096, 7, 7)
            del select_features, glob_feature
            x = self.relu1(self.conv1(x)) # shape(1, 512, 5, 5)
            x = self.avgpool(x) # shape(1, 512, 1, 1)
            x = x.reshape(1, -1)
            x = self.fc1_relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)




        return x






class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.conv = nn.Conv2d(1024, 1, 14)

    def forward(self, x):
        # x.shape (N, 1024, 14, 14)
        weights = self.conv(x) # weights.shape (N, 1024, 1, 1)

        return weights
