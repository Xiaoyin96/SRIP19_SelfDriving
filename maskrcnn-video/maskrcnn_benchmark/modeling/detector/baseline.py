# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import logging
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from .predictor2 import build_predictor # 4 x 4


# def DrawBbox(img, boxlist):
#     plt.imshow(img)
#     currentAxis = plt.gca()
#     # folder = '/data6/SRIP19_SelfDriving/bdd12k/Outputs/'
#     for i in range(boxlist.shape[0]):
#         bbox = boxlist[i]
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
#         # rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='r', facecolor='none')
#         currentAxis.add_patch(rect)
#
#     plt.show()


class Baseline(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(Baseline, self).__init__()

        self.training = False
        # self.backbone = build_backbone(cfg)
        # self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, 1024)
        self.roi_heads.box.feature_extractor.head.load_state_dict(torch.load('/home/selfdriving/maskrcnn-benchmark/configs/weights/layer4.pth'))  # pre-trained weights
        self.predictor = build_predictor(cfg)
        


    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features: from I3D backbone
            targets: ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        x, result, _ = self.roi_heads([features], [proposals], [proposals]) # should be list
        xx = {}
        xx['glob_feature'] = features # 1 x 1024 x 14 x 14
        xx['roi_features'] = x # num of bbox x 2048 x 7 x 7 / 4 x 4
        # print(x.shape)
        pred = self.predictor(xx) # torch.size([4])
        return pred




