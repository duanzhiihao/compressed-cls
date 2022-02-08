import torch
import torch.nn as nn


class DummyTeacher(nn.Identity):
    def __init__(self, feature_num=4):
        super().__init__()
        self.cache = [None for _ in range(feature_num)]

    def forward(self, input):
        return None


class ResNetTeacher(nn.Module):
    def __init__(self, source='torchvision'):
        super().__init__()
        if source == 'torchvision':
            from torchvision.models.resnet import resnet50
        elif source == 'timm':
            from timm.models.resnet import resnet50
        else:
            raise ValueError()
        _model = resnet50(pretrained=True)
        self.conv1 = _model.conv1
        self.bn1   = _model.bn1
        self.act1  = _model.act1
        self.maxpool = _model.maxpool
        self.layer1 = _model.layer1
        self.layer2 = _model.layer2
        self.layer3 = _model.layer3
        self.layer4 = _model.layer4
        self.global_pool = _model.global_pool
        self.fc = _model.fc
        self.cache = [None, None, None, None]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.global_pool(x4)
        x = self.fc(x)
        self.cache = [x1, x2, x3, x4]
        return x


def main():
    
    debug = 1


if __name__ == '__main__':
    main()
