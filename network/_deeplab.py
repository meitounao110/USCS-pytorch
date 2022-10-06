import torch
from torch import nn
from torch.nn import functional as F
# from collections import OrderedDict
from .utils import _SimpleSegmentationModel
from .backbone import resnet_v1

# from utils.mask import grid_mask

# import torchvision.transforms.functional as TF
# from network.noise import FeatureDropDecoder, FeatureNoiseDecoder, DropOutDecoder

# try:  # for torchvision<0.4
#     from torchvision.models.utils import load_state_dict_from_url
# except:  # for torchvision>=0.4
#     from torch.hub import load_state_dict_from_url
#
__all__ = ["DeepLabHeadV3Plus", "DeepLabV3mimov2"]


def mask_produce(feature):
    device = feature.device
    mask = torch.zeros(feature.shape)
    mask[..., 0::2, 1::2] = 1
    mask[..., 1::2, 0::2] = 1
    mask1 = 1 - mask
    return mask.to(device), mask1.to(device)


class DeepLabV3mimov2(nn.Module):
    def __init__(self, backbone, classifier, num_classes, num_members, bn_momentum):
        super(DeepLabV3mimov2, self).__init__()
        self.num_members = num_members
        self.backbone = backbone
        self.classifier = classifier
        self._init_final_layer(num_members, num_classes, bn_momentum)

    def forward(self, x, batch_size=1):
        input_shape = x.shape[-2:]
        x, low_level_feat = self._forward_backbone(x, self.num_members, batch_size)
        x = self.classifier(x, low_level_feat)
        x = self._forward_final_layer(x, input_shape, self.num_members)
        return x

    def _make_final(self, num_classes, bn_momentum):
        classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )
        return classifier

    def _init_final_layer(self, num_members, num_classes, bn_momentum):
        """
        Initialize the M output heads/classifiers
        """
        list_final = []
        for _ in range(0, num_members):
            list_final.append(self._make_final(num_classes, bn_momentum))
        self.final = nn.ModuleList(list_final)
        self._init_weight(self.final)

    def _forward_backbone(self, pixels, num_members, batch_size):
        if num_members == 1 or pixels.size(0) == batch_size:
            return self.backbone(pixels)
        else:
            # pixels_member = torch.cat(torch.split(pixels, 3, dim=1), dim=0)
            block = self.backbone(pixels)
            list_lfeats = torch.chunk(block[0], chunks=2, dim=0)
            list_lfeats_low = torch.chunk(block[1], chunks=2, dim=0)
            mask_low, mask1_low = mask_produce(list_lfeats_low[0])
            mask, mask1 = mask_produce(list_lfeats[0])
            aggreg_lowlevel = mask_low * list_lfeats_low[0] + mask1_low * list_lfeats_low[1]
            aggreg_out = mask * list_lfeats[0] + mask1 * list_lfeats[1]
            # tensor_aggreg = 0.5 * list_lfeats[0] + 0.5 * list_lfeats[1]
            return aggreg_out, aggreg_lowlevel

    def _forward_final_layer(self, extracted_features, input_shape, num_members):
        dict_output = {}
        # compute individual x
        for num_member in range(0, num_members):
            x_n = self.final[num_member](extracted_features)
            x_n = F.interpolate(x_n, size=input_shape, mode='bilinear', align_corners=True)
            dict_output["final_" + str(num_member)] = x_n

        # compute ensemble x by averaging
        _list_x = [
            dict_output["final_" + str(num_member)]
            for num_member in range(0, num_members)
        ]
        dict_output["final"] = torch.stack(_list_x, dim=0).mean(dim=0)
        return dict_output

    def _init_weight(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 modeling from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the modeling.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, num_classes, bn_momentum=0.1, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(aspp_dilate, bn_momentum)

        # self.classifier = nn.Sequential(
        #     nn.Conv2d(304, 256, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(256, momentum=bn_momentum),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     # nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #     # nn.BatchNorm2d(256, momentum=bn_momentum),
        #     # nn.ReLU(inplace=True),
        #     # nn.Dropout(0.1),
        #     nn.Conv2d(256, num_classes, 1)
        # )

        self._init_weight()

    def forward(self, feature, low_level_feat, step=2):
        low_level_feature = self.project(low_level_feat)
        output_feature = self.aspp(feature)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=True)
        output = torch.cat([low_level_feature, output_feature], dim=1)

        # if step == 1:
        #     random_feature = np.random.choice(self.random_feature)
        #     output = random_feature(output)

        # output = self.classifier(output)
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, bn_momentum):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, bn_momentum):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, atrous_rates, bn_momentum=0.1):
        super(ASPP, self).__init__()
        in_channels = 2048
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, bn_momentum))
        modules.append(ASPPConv(in_channels, out_channels, rate2, bn_momentum))
        modules.append(ASPPConv(in_channels, out_channels, rate3, bn_momentum))
        modules.append(ASPPPooling(in_channels, out_channels, bn_momentum))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
