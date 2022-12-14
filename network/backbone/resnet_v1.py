# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 17:45
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : ResNet101.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50stem': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101stem': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, bn_momentum=0.1, pretrained=False, output_stride=16, deep_stem=True):
        if output_stride == 16:
            dilations = [1, 1, 1, 2]
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4]
            strides = [1, 2, 1, 1]
        else:
            raise Warning("output_stride must be 8 or 16!")
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.deep_stem = deep_stem
        if deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes // 2, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes // 2, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
                                       bn_momentum=bn_momentum)
        self.fc = nn.Linear(512 * block.expansion, 1000)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()
        del self.fc, # self.stem, self.maxpool

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum),
            )

        layers = []
        previous_dilation = dilation
        # if dilation != 1:
        #     previous_dilation = 1
        layers.append(block(self.inplanes, planes, stride, previous_dilation, downsample, bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if dilation != 1:
                dilation *= 2
            layers.append(block(self.inplanes, planes, dilation=dilation, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet50stem'])
        if "state_dict" in pretrain_dict:
            pretrain_dict = pretrain_dict["state_dict"]
        # model_dict = {}
        # state_dict = self.state_dict()
        # for k, v in pretrain_dict.items():
        #     if k in state_dict:
        #         model_dict[k] = v
        # state_dict.update(model_dict)
        # self.load_state_dict(state_dict)
        self.load_state_dict(pretrain_dict)
        print("Having loaded imagenet-pretrained successfully!")

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def resnet101(bn_momentum=0.1, pretrained=True, output_stride=16):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], bn_momentum, pretrained, output_stride)
    return model


def resnet50(bn_momentum=0.1, pretrained=True, output_stride=16):
    """Constructs a ResNet-50-v1 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], bn_momentum, pretrained, output_stride)
    return model


if __name__ == "__main__":
    model = resnet50(pretrained=True)
    model.eval()
    print(model)
    x = torch.randn(16, 3, 513, 513)
    out = model(x)
    print(out[0].shape)
