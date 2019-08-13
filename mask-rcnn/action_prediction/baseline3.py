import torch
import torch.nn as nn
from maskrcnn_benchmark.modeling.backbone.resnet import ResNetHead
from maskrcnn_benchmark.modeling.backbone import resnet

class baseline(nn.Module):
    """
    This baseline simply add/cat outputs of ROI
    roi_num = # proposals
    class_num = # actions
    """

    def __init__(self, config, class_num=4, is_cat=False, is_feature=False):
        super(baseline, self).__init__()
        self.class_num = class_num  # number of actions
        self.is_cat = is_cat
        self.is_feature = is_feature
        
        # conv layers
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.head = ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION) # TODO: specify the arguments here! output: (1, 2048, 14, 14)
        self.head.load_state_dict(torch.load('/data6/SRIP19_SelfDriving/Outputs/layer4.pth'))
        self.selector = Selector()
        self.softmax = nn.Softmax(dim=0)


        if not self.is_cat:
            self.avgpool1 = nn.AdaptiveAvgPool3d(output_size=1)
            self.conv1 = nn.Conv2d(2048+1024, 512, 3)
            self.relu1 = nn.PReLU(512)
            #self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=1)
            self.drop = nn.Dropout(p=0.5)
            self.fc1 = nn.Linear(512, 512)
            self.relu_1 = nn.PReLU(512)
            self.fc2 = nn.Linear(512, 4)
        else:
            self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1)
            self.conv1 = nn.Conv2d(2048, 512, 1)
            self.relu1 = nn.PReLU(512)
            self.drop = nn.Dropout(p=0.5)
            self.fc1 = nn.Linear(512*6, 512)
            self.relu_1 = nn.PReLU(512)
            self.fc2 = nn.Linear(512, 4)

        
        

    def forward(self, x):
        """
        x (List) is the list of features from ROI pooling/align with global feature
        extracted from backbone. 
        e.g. x[0].size = (N + 1, 1024, 14, 14),
        N is # objects, 1 means the avgpooled global feature.
        """
        # obj, glob = self.DataPreprocess(x) # Deal with ROI pooled features
                
        # 
        # x = torch.cat(obj, dim=0) # x.shape = (Batchsize, 1024*3, 14, 14) if cat=True
                                    # x.shape = (Batchsize, 1024, 14, 14) if cat=False
        
        obj = x[:-1] # shape (N, 1024, 14, 14)
        glo = x[-1] # shape(1, 1024, 14, 14)
        
        scores = self.selector(obj)
        scores, idx = torch.sort(scores, dim=0, descending=True)
        scores_logits = self.softmax(scores)
        idx = idx[:5].reshape(5)
        obj = obj[idx] # shape (5, 1024, 14, 14)
        obj = scores_logits[:5] * obj # shape (5, 1024, 14, 14)

        if not self.is_cat:
            obj = torch.mean(obj, dim=0) # shape(1, 1024, 14, 14)
            x = torch.cat((obj, glo.unsqueeze(0)), dim=0) # shape(2, 1024, 14 ,14)
            x = self.head(x) # shape(2, 2048, 7, 7)
            x = self.conv1(x) # shape(2, 512,)
            x = self.relu1(x) 
            x = obj.transpose(1, 0) # shape(512, 2, )
            x = self.avgpool1(x) # shape(512, 1, 1, 1)
            x = x.reshape(1, -1) # shape(1, 512)
            x = self.fc1(x) # shape(1, 512)
            x = self.relu_1(x)
            x = self.drop(x)
            pred = self.fc2(x) # shape(1, 4)            
            
        else:
            # print("obj:", obj.shape)
            # print("glo:", glo.shape)
            x = torch.cat((obj, glo.unsqueeze(0)), dim=0) # shape(6, 1024, 14 ,14)
            x = self.head(x) # shape(6, 2048, 7, 7)
            x = self.conv1(x) # shape (6, 512, 5, 5)
            # print(x.shape)
            x = self.relu1(x)
            x = self.avgpool1(x) # shape (6, 512, 1, 1)
            x = x.reshape(1, 6*512) # shape(1, 512*6, 1, 1)
            x = self.fc1(x)
            # print(x.shape)
            # print(self.relu_1)
            x = self.relu_1(x) # shape(1, 512*6, 1, 1)
            x = self.drop(x)
            pred = self.fc2(x) # shape(1, 4)

                           
        return pred

    def DataPreprocess(self, x):
        """
        x (List) should be a list of Tensors.
        e.g. x[0] = Tensor(N + 1, 1024, 14, 14),
        """
        n = 3
        obj = []
        glob = []
        for i, feature in enumerate(x):
            # normalize features
            x[i] = torch.div(feature, feature.norm())            
            if not self.is_cat:
                ROI_sum = torch.sum(x[i][:-1], dim=0).unsqueeze(0) # out.shape = (1, 1024, 14, 14)
                obj.append(ROI_sum)
                glob.append(x[i][-1].unsqueeze(0))
            else:
                ROI_cat = torch.cat(x[i][:n], dim=1) #TODO: figure out how to concatenate exactly
                obj.append(ROI_cat) # each element in the list is tensor with size = (1, 1024*3, 14, 14)
                glob.append(x[i][-1].unsqueeze(0))
                
        
        return obj, glob

class Selector(nn.Module):
    """
    Selector is used to create sparse object attetion.
    """
    def __init__(self):
        super(Selector, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1, 14)
        
    def forward(self, x):
        """
        x(Tensor) is ROI-Pooled features with shape e.g. (N, 1024, 14, 14) 
        """
        weights = self.conv1(x)
        
        return weights
        
    

def main():
    """
    Just a test.
    """
    x = torch.ones(1000, 256, 7, 7)
    net = baseline()
    out = net(x)
    print(out)
    print(out.shape)


if __name__ == "__main__":
    main()
