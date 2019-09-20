# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn
import torchvision

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import i3res_rectangle
import copy

@registry.BACKBONES.register("R-50-I3D-C4")
def build_resnet_i3d_backbone(cfg):
    choice = cfg.MODEL.I3D_RESNET.LAYERS
    if choice == 50:
        print('load resnet50 pretrained model...')
        resnet_2d = torchvision.models.resnet50(pretrained=True)        
    elif choice == 101:
        print('load resnet101 pretrained model...')
        resnet_2d = torchvision.models.resnet101(pretrained=True)       
    elif choice == 152:
        print('load resnet152 pretrained model...')
        resnet_2d = torchvision.models.resnet152(pretrained=True)    
    else:
        raise ValueError('resnet_nb should be in [50|101|152] but got {}'
                         ).format(choice)
    model = i3res_rectangle.I3ResNet(copy.deepcopy(resnet_2d), 
                                    cfg.MODEL.I3D_RESNET.FRAME_NB, 
                                    cfg.MODEL.I3D_RESNET.CLASS_NB, 
                                    cfg.MODEL.I3D_RESNET.REASON_NB, 
                                    cfg.MODEL.I3D_RESNET.SIDE_TASK, 
                                    cfg.MODEL.I3D_RESNET.CONV_CLASS,
                                    cfg.MODEL.I3D_RESNET.RETURN_FEATURE)
    
    return model



@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
