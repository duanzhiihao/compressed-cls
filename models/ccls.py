import torch
import torch.nn as nn
import torch.nn.functional as tnf

from compressai.zoo import cheng2020_anchor

from models.resnet import Bottleneck, conv1x1
from my_utils import load_partial
from datasets import IMAGENET_MEAN, IMAGENET_STD


class VCMClassify(nn.Module):
    """
    docstring
    """
    def __init__(self, stage1='', stage2='', stage3='', num_cls=1000, verbose=True):
        super().__init__()

        self.need_quant = True
        if stage1.startswith('cheng'):
            if stage1 in ('cheng1', 'cheng3'):
                ch1 = 128
            elif stage1 in ('cheng5'):
                ch1 = 192
            else:
                raise ValueError()

            q = int(stage1[-1])
            _model = cheng2020_anchor(q, metric="mse", pretrained=True)
            self.stage1 = _model.g_a
        elif stage1 == '':
            self.stage1 = nn.Identity()
            ch1 = None
            self.need_quant = False
        else:
            raise NotImplementedError()
        # disable training
        for v in self.stage1.parameters():
            v.requires_grad = False

        self.need_center = False
        if stage2.startswith('cheng'):
            q = int(stage2[-1])
            _model = cheng2020_anchor(q, metric="mse", pretrained=True)
            self.stage2 = _model.g_s
            self.need_center = True
        elif stage2 == '':
            self.stage2 = torch.nn.Identity()
        else:
            raise NotImplementedError()
        # disable training
        for v in self.stage2.parameters():
            v.requires_grad = False

        if stage3 == 'res50-aa':
            self.stage3 = ResNet_aa(ch1, [3, 4, 6, 3], num_classes=num_cls)
            from models.resnet import model_urls
            msd = torch.load(model_urls['resnet50'], weights_only=True)
            load_partial(self.stage3, msd, verbose=verbose)
        elif stage3 == 'res50':
            from models.resnet import resnet50
            self.stage3 = resnet50(num_classes=num_cls, pretrained=True)
        elif stage3 == '':
            self.stage3 = torch.nn.Identity()
        else:
            raise NotImplementedError()

        self.input_mean = torch.FloatTensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.input_std  = torch.FloatTensor(IMAGENET_STD).view(1, 3, 1, 1)

    def forward(self, x: torch.FloatTensor, _return_cache=False):
        assert x.dim() == 4 and (not x.requires_grad)

        with torch.no_grad():
            x = self.stage1(x)
            if self.need_quant:
                x = torch.round(x)

            x = self.stage2(x)
            if self.need_center:
                self.input_mean = self.input_mean.to(device=x.device)
                self.input_std = self.input_std.to(device=x.device)
                x.sub_(self.input_mean).div_(self.input_std)

        x = self.stage3(x, _return_cache)
        return x


class ResNet_aa(nn.Module):
    def __init__(self, in_ch=192, layers=[3, 4, 6, 3], num_classes=1000):
        super().__init__()
        block = Bottleneck
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.act = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, self.inplanes*16, kernel_size=3, padding=1),
            nn.PixelShuffle(4)
        )
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
        # assert x.shape[2:4] == (14,14)
        # x = tnf.interpolate(x, scale_factor=(4,4), mode='bilinear', align_corners=False)
        x = self.upsample(x)
        # x: [b, 3, H, W]
        # x = self.conv1(x) # x: [b, 64, H/2, W/2]
        # x = self.bn1(x)
        # x = self.act(x)

        x1 = self.layer1(x) # x: [b, 256, H/4, W/4]
        x2 = self.layer2(x1) # x: [b, 512, H/8, W/8]
        x3 = self.layer3(x2) # x: [b, 1024, H/16, W/16]
        x4 = self.layer4(x3) # x: [b, 2048, H/32, W/32]

        x = tnf.adaptive_avg_pool2d(x4, output_size=(1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        # x: [b, num_class]
        if _return_cache:
            return x, x1, x2, x3, x4
        return x
