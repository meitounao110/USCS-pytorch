from ._deeplab import DeepLabHeadV3Plus, DeepLabV3mimov2
from .backbone import resnet_v1
import torch
import torch.nn as nn


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]
    backbone = resnet_v1.__dict__[backbone_name](pretrained=pretrained_backbone, output_stride=output_stride)
    # backbone = resnet.__dict__[backbone_name](
    #     pretrained=pretrained_backbone,
    #     replace_stride_with_dilation=replace_stride_with_dilation)

    if name == 'deeplabv3plus':
        classifier = DeepLabHeadV3Plus(num_classes, bn_momentum=0.1, aspp_dilate=aspp_dilate)
    # model = DeepLabV3(backbone, classifier)
    # model = DeepLabV3mimo(backbone, classifier, num_classes, bn_momentum=0.1)
    # model = DeepLabV3mimov1(backbone, num_classes, aspp_dilate)
    model = DeepLabV3mimov2(backbone, classifier, num_classes, num_members=2, bn_momentum=0.1)
    return model


# Deeplab v3+
def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ modeling with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """

    return _segm_resnet('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride,
                        pretrained_backbone=pretrained_backbone)
    # return Network('resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ modeling with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _segm_resnet('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                        pretrained_backbone=pretrained_backbone)
