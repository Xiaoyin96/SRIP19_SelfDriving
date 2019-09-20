# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .baseline import Baseline
from .generalized_rcnn import GeneralizedRCNN
from .generalized_model import GeneralizedModel
# from .ns import NS


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, 'Baseline':Baseline, 'GeneralizedModel':GeneralizedModel}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
