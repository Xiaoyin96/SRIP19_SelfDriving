import torch
import torch.nn as nn

from maskrcnn_benchmark.modeling.detector import build_detection_model


class Network(nn.Module):

    def __init__(self, config, select_num=5, class_num=5, is_cat=True):
        super(Network, self).__init__()
        self.detector = build_detection_model(config)
        self.detector.eval()
        self.selector = Selector()
        self.select_num = select_num
        self.sftmax = nn.Softmax()
        self.is_cat = is_cat
        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=14)
        self.conv_glob = nn.Conv2d(1024, 2048, 2, stride=2)
        self.relu_glob = nn.PReLU(2048)
        if self.is_cat:
            self.conv1 = nn.Conv2d(2048, 512, 1)
            self.relu1 = nn.PReLU(512)
            self.fc1 = nn.Linear(512*self.select_num, 512) # select top 5
            self.fc1_relu = nn.PReLU(512)
            self.fc2 = nn.Linear(512, class_num)
        else:
            self.conv1 = nn.Conv2d(4096, 512, 1, 1)
            self.relu1 = nn.PReLU(512)
            self.fc1 = nn.Linear(512)
            self.fc1_relu = nn.PReLU(512)
            self.fc2 = nn.Linear(512, class_num)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout()



    def forward(self, x):
        detect_results = self.detector(x) #TODO: decide detector results


        scores = self.selector(detect_results['bottleneck_features'])
        scores, idx = torch.sort(scores, dim=0, descending=True)
        scores_logits = self.sftmax(scores)
        idx = idx[:self.select_num].reshape(self.select_num)
        select_features = detect_results['bottleneck_features'][idx]
        select_features *= scores_logits[:self.select_num] # shape(select_num, 2048, 7, 7)

        glob_feature = self.avgpool_glob(detect_results['glob_feature']) # shape(1, 1024, 14, 14)
        glob_feature = self.relu_glob(self.conv_glob(glob_feature)) # shape(1, 2048, 7, 7)


        if self.is_cat:
            x = torch.cat((select_features, glob_feature), dim=0) # shape(select_num+1, 2048, 7, 7)
            x = self.relu1(self.conv1(x)) # shape(select_num+1, 512, 5, 5)
            x = self.avgpool(x) # shape(select_num+1, 512, 1, 1)
            x = x.reshape(1, -1) # shape(1, (select_num+1)*512)
            x = self.fc1_relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

        else:
            select_features = torch.sum(select_features, dim=0) # shape(1, 2048, 7, 7)
            x = torch.cat((select_features, glob_feature), dim=0) # shape(1, 4096, 7, 7)
            x = self.relu1(self.conv1(x)) # shape(1, 512, 5, 5)
            x = self.avgpool(x) # shape(1, 512, 1, 1)
            x = self.fc1_relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2




        return x






class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.conv = nn.Conv2d(1024, 1, 14)

    def forward(self, x):
        # x.shape (N, 1024, 14, 14)
        weights = self.conv(x) # weights.shape (N, 1024, 1, 1)

        return weights