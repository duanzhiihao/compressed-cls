import torch
import torch.nn as nn
import torch.nn.functional as tnf


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    '''
    Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    '''
    expansion = 4
    def __init__(self, inplanes, hidden, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, hidden)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = conv3x3(hidden, hidden, stride)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = conv1x1(hidden, hidden * self.expansion)
        self.bn3 = nn.BatchNorm2d(hidden * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


class ResNet(nn.Module):
    '''
    ResNet from torchvision
    '''
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, _return_cache=False):
        # x: [b, 3, H, W]
        x = self.conv1(x) # x: [b, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.act(x)
        # x = self.maxpool(x) # x: [b, 64, H/4, W/4]
        x = tnf.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x1 = self.layer1(x) # x: [b, 256, H/4, W/4]
        x2 = self.layer2(x1) # x: [b, 512, H/8, W/8]
        x3 = self.layer3(x2) # x: [b, 1024, H/16, W/16]
        x4 = self.layer4(x3) # x: [b, 2048, H/32, W/32]

        # x = self.avgpool(x4) # x: [b, 2048, 1, 1]
        x = tnf.adaptive_avg_pool2d(x4, output_size=(1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        self.cache = [x1, x2, x3, x4]
        # x: [b, num_class]
        if _return_cache:
            return x, x1, x2, x3, x4
        return x


def resnet50(num_classes, pretrained=True):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        msd = torch.hub.load_state_dict_from_url(model_urls['resnet50'], progress=True)
        model.load_state_dict(msd)
    return model


def resnet101(num_classes):
    """ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model


def resnet152(num_classes):
    """ResNet-152 model from
    Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model
