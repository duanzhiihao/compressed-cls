import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision as tv

from mycv.paths import MYCV_DIR
from mycv.utils.torch_utils import load_partial, weights_replace


class VCMClassify(nn.Module):
    """
    docstring
    """
    def __init__(self, stage1='', stage2='', stage3='', num_cls=1000,
                 verbose=True):
        super().__init__()
        self.need_quant = True
        if stage1 == 'miniNIC':
            raise DeprecationWarning()
            from mycv.models.nlaic.mini import miniEnc
            self.stage1 = miniEnc(input_ch=3, hyper=False)
            weights = weights_replace('weights/mini6_last.pt', 'encoder.', '')
            load_partial(model.stage1, weights, verbose=True)
            # weights = weights_replace('weights/mini6_last.pt', 'decoder.', '')
            # load_partial(model.stage2, weights)
            ch1 = 192
        elif stage1.startswith('nlaic'):
            from mycv.models.nlaic.nlaic import Enc
            self.stage1 = Enc(enable_hyper=False)
            weights = weights_replace(MYCV_DIR / f'weights/nlaic/{stage1}.pt', 'encoder.', '')
            load_partial(self.stage1, weights, verbose=verbose)
            ch1 = 192
        elif stage1.startswith('cheng'):
            if stage1 in ('cheng1', 'cheng3'):
                ch1 = 128
            elif stage1 in ('cheng5'):
                ch1 = 192
            else:
                raise ValueError()
            from mycv.external.compressai.zoo import cheng2020_anchor
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

        self.need_center = True
        if stage2 == 'miniDec':
            raise DeprecationWarning()
            from mycv.models.nlaic.mini import miniDec
            self.stage2 = miniDec(input_features=3)
        elif stage2.startswith('nlaic'):
            from mycv.models.nlaic.nlaic import Dec
            self.stage2 = Dec()
            weights = weights_replace(MYCV_DIR / f'weights/nlaic/{stage2}.pt', 'decoder.', '')
            load_partial(self.stage2, weights, verbose=verbose)
        elif stage2.startswith('cheng'):
            from mycv.external.compressai.zoo import cheng2020_anchor
            q = int(stage2[-1])
            _model = cheng2020_anchor(q, metric="mse", pretrained=True)
            self.stage2 = _model.g_s
        elif stage2 == '':
            self.stage2 = torch.nn.Identity()
            self.need_center = False
        else:
            raise NotImplementedError()
        # disable training
        for v in self.stage2.parameters():
            v.requires_grad = False

        self._flops_mode = False
        # if stage3 == 'res50_':
        #     raise DeprecationWarning()
        #     self.stage3 = ResNet_(ch1, [3, 4, 6, 3], num_classes=num_cls, student=student)
        # elif stage3 == 'aares50':
        #     self.stage3 = aaResNet(ch1, [3, 4, 6, 3], num_classes=num_cls)
        #     wpath = MYCV_DIR / 'weights/resnet/resnet50-19c8e357.pth'
        #     load_partial(self.stage3, wpath, verbose=verbose)
        # el
        if stage3 == 'res50-aa-spc':
            self.stage3 = ResNet_aa_spc(ch1, [3, 4, 6, 3], num_classes=num_cls)
            wpath = MYCV_DIR / 'weights/resnet/resnet50-19c8e357.pth'
            load_partial(self.stage3, wpath, verbose=verbose)
        elif stage3 == 'reshape-res50':
            self.stage3 = ReshapeResNet50(ch1, num_classes=num_cls)
        elif stage3 == 'cres50':
            self.stage3 = cResNet(ch1, [0, 4, 10, 3], num_classes=num_cls)
            wpath = MYCV_DIR / 'weights/resnet/resnet50-19c8e357.pth'
            load_partial(self.stage3, wpath, verbose=verbose)
        elif stage3 == 'res50':
            from mycv.models.cls.resnet import resnet50
            self.stage3 = resnet50(num_classes=num_cls, pretrained=True)
        elif stage3 == 'efb0-aa':
            assert num_cls == 1000
            # self.stage3 = effnet_b0_aa(in_ch=ch1)
            self.stage3 = EffNet_b0_aa(in_ch=ch1)
            wpath = MYCV_DIR / 'weights/efficientnet/efb0_7.pt'
            load_partial(self.stage3, wpath, verbose=verbose)
        elif stage3 == 'efb0':
            assert num_cls == 1000
            # from mycv.external.timm.efficientnet import efficientnet_b0_wrapper
            # self.stage3 = efficientnet_b0_wrapper()
            from mycv.external.timm.efficientnet import EfficientNetWrapper
            self.stage3 = EfficientNetWrapper('b0')
            wpath = MYCV_DIR / 'weights/efficientnet/efb0_7.pt'
            load_partial(self.stage3, wpath, verbose=verbose)
        elif stage3 == 'vgg11-aa':
            self.stage3 = VGG11aa(ch1, num_classes=num_cls)
            wpath = MYCV_DIR / 'weights/vgg/vgg11-tv.pth'
            load_partial(self.stage3, wpath, verbose=verbose)
        elif stage3 == 'vgg11':
            # from mycv.models.cls.vgg import VGG
            # self.stage3 = VGG(version='vgg11')
            # wpath = MYCV_DIR / 'weights/vgg/vgg11-tv.pth'
            # load_partial(self.stage3, wpath, verbose=verbose)
            from torchvision.models.vgg import vgg11
            self.stage3 = vgg11(pretrained=True)
            self._flops_mode = True
        # elif stage3.startswith('csp_'):
        #     self.stage3 = CSP_(model=stage3[-1], in_ch=128, num_class=num_cls)
        elif stage3 == 'vitt-aa':
            self.stage3 = ViTaa(ch1, num_classes=num_cls)
        elif stage3 == 'vitt':
            from timm.models.vision_transformer import vit_tiny_patch16_224
            self.stage3 = vit_tiny_patch16_224(pretrained=True)
        elif stage3 == '':
            self.stage3 = torch.nn.Identity()
        else:
            raise NotImplementedError()

        from mycv.datasets.constants import IMAGENET_MEAN, IMAGENET_STD
        self.input_mean = torch.FloatTensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.input_std  = torch.FloatTensor(IMAGENET_STD).view(1, 3, 1, 1)

    def forward(self, x: torch.FloatTensor, _return_cache=False):
        assert x.dim() == 4 and (not x.requires_grad)

        with torch.no_grad():
            x = self.stage1(x)
            if isinstance(x, tuple):
                assert len(x) == 2 and x[1] is None
                x = x[0]
            if self.need_quant:
                x = torch.round(x)
            x = self.stage2(x)
            if self.need_center:
                self.input_mean = self.input_mean.to(device=x.device)
                self.input_std = self.input_std.to(device=x.device)
                x.sub_(self.input_mean).div_(self.input_std)

        if self._flops_mode:
            x = self.stage3(x)
        else:
            x = self.stage3(x, _return_cache)
        return x

    def forward_nic(self, imgs):
        assert not self.training
        x1 = self.stage1(imgs)
        if isinstance(x1, tuple):
            x1, _ = x1
        x1 = torch.round(x1)
        x2 = self.stage2(x1)
        x2 = x2.clamp_(min=0, max=1)
        return x2, None


class ReshapeResNet50(nn.Module):
    def __init__(self, in_ch=192, num_classes=1000):
        super().__init__()
        assert num_classes == 1000
        self.resnet50 = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)

    def forward(self, x, _return_cache=False):
        assert x.shape[1] % 3 == 0
        factor = round(math.sqrt(x.shape[1] / 3))
        assert isinstance(factor, int)
        tnf.pixel_shuffle(x, upscale_factor=factor)
        assert (x.shape[1] == 3) and (x.shape[2] == x.shape[3])
        y = self.resnet50(x)
        return y


from mycv.models.cls.resnet import Bottleneck, conv1x1
class aaResNet(nn.Module):
    '''
    ResNet from torchvision
    '''
    def __init__(self, in_ch=192, layers=[3, 4, 6, 3], num_classes=1000):
        super().__init__()
        block = Bottleneck
        self.inplanes = in_ch
        # self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.act = nn.ReLU(inplace=True)
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
        x = tnf.interpolate(x, scale_factor=(4,4), mode='bilinear', align_corners=False)
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
        self.cache = [x1, x2, x3, x4]
        if _return_cache:
            raise NotImplementedError()
        return x

class ResNet_aa_spc(nn.Module):
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


class VGG11aa(nn.Module):
    def __init__(self, in_ch=192, num_classes=1000):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(in_ch, 64*16, kernel_size=3, padding=1),
            nn.PixelShuffle(4)
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, _return_cache=False):
        # x = tnf.interpolate(x, scale_factor=(4,4), mode='bilinear', align_corners=False)
        x1 = self.m1(x)
        x2 = self.m2(x1)
        x3 = self.m3(x2)
        x4 = self.m4(x3)
        x5 = self.m5(x4)
        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if _return_cache:
            return x, None, x3, x4, x5
        return x


def convrelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    m = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True)
    )
    return m


class VGG11Teacher(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        block_num = [1, 1, 2, 2, 2]
        self.m1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        wpath = MYCV_DIR / 'weights/vgg/vgg11-tv.pth'
        msd = torch.load(wpath)
        self.load_state_dict(msd)

    def forward(self, x, _return_cache=False):
        x1 = self.m1(x)
        x2 = self.m2(x1)
        x3 = self.m3(x2)
        x4 = self.m4(x3)
        x5 = self.m5(x4)
        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if _return_cache:
            return x, x2, x3, x4, x5
        return x


def _forward(self, x, _return_cache=False):
    imh, imw = x.shape[2:4]

    x = self.upsample(x)
    features = []
    prev_x = x
    for block in self.blocks:
        x = block(x)
        if prev_x.size(2) > x.size(2):
            features.append(prev_x)
        prev_x = x
    x = self.conv_head(x)
    features.append(x)
    x = self.bn2(x)
    x = self.act2(x)
    x = self.global_pool(x)
    if self.drop_rate > 0.:
        x = tnf.dropout(x, p=self.drop_rate, training=self.training)
    yhat = self.classifier(x)

    if _return_cache:
        x2, x3, x4, x5 = features
        return yhat, x2, x3, x4, x5
    else:
        return yhat


def effnet_b0_aa(in_ch=192, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    from timm.models.efficientnet import efficientnet_b0, EfficientNet
    model = efficientnet_b0(pretrained=True, **kwargs)
    model: EfficientNet
    del model.conv_stem, model.bn1, model.act1
    model.upsample = nn.Sequential(
        nn.Conv2d(in_ch, 32*16, kernel_size=3, padding=1),
        nn.PixelShuffle(4)
    )
    model.blocks[0] = nn.Identity()
    model.blocks[1][0].conv_pw = nn.Conv2d(32, 96, kernel_size=1, padding=0)
    model.blocks[1][0].conv_dw.stride = (1,1)
    model.forward = types.MethodType(_forward, model)
    return model


class EffNet_b0_aa(nn.Module):
    def __init__(self, in_ch=192, **kwargs):
        super().__init__()
        from timm.models.efficientnet import efficientnet_b0, EfficientNet
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, 32*16, kernel_size=3, padding=1),
            nn.PixelShuffle(4)
        )
        model = efficientnet_b0(pretrained=True, **kwargs)
        model.blocks[0] = nn.Identity()
        model.blocks[1][0].conv_pw = nn.Conv2d(32, 96, kernel_size=1, padding=0)
        model.blocks[1][0].conv_dw.stride = (1,1)
        self.blocks = model.blocks
        self.conv_head   = model.conv_head
        self.bn2         = model.bn2
        self.act2        = model.act2
        self.global_pool = model.global_pool
        self.drop_rate   = model.drop_rate
        self.classifier  = model.classifier

    def forward(self, x, _return_cache=False):
        imh, imw = x.shape[2:4]

        x = self.upsample(x)
        features = []
        prev_x = x
        for block in self.blocks:
            x = block(x)
            if prev_x.size(2) > x.size(2):
                features.append(prev_x)
            prev_x = x
        x = self.conv_head(x)
        features.append(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = tnf.dropout(x, p=self.drop_rate, training=self.training)
        yhat = self.classifier(x)

        if _return_cache:
            x2, x3, x4, x5 = features
            return yhat, x2, x3, x4, x5
        else:
            return yhat


class cResNet(nn.Module):
    '''
    ResNet from torchvision
    '''
    def __init__(self, in_ch, layers=[0, 4, 10, 3], num_classes=1000):
        super().__init__()
        assert len(layers) == 4 and layers[0] == 0
        block = Bottleneck
        self.inplanes = in_ch
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
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
        # x1 = self.layer1(x) # x: [b, 256, H/4, W/4]
        x2 = self.layer2(x) # x: [b, 512, H/8, W/8]
        x3 = self.layer3(x2) # x: [b, 1024, H/16, W/16]
        x4 = self.layer4(x3) # x: [b, 2048, H/32, W/32]

        # x = self.avgpool(x4) # x: [b, 2048, 1, 1]
        x = tnf.adaptive_avg_pool2d(x4, output_size=(1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        self.cache = [None, None, x3, x4]
        # x: [b, num_class]
        if _return_cache:
            return x, None, None, x3, x4
        return x


class ViTaa(nn.Module):
    def __init__(self, in_ch=192, num_classes=1000):
        super().__init__()
        from timm.models.vision_transformer import vit_tiny_patch16_224
        model = vit_tiny_patch16_224(pretrained=True)
        assert model.embed_dim == in_ch and model.num_classes == num_classes
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        assert model.dist_token is None
        self.pos_drop = model.pos_drop
        self.blocks = model.blocks
        self.norm = model.norm
        assert isinstance(model.pre_logits, nn.Identity)
        assert model.head_dist is None
        self.head = model.head

    def forward(self, x, _return_cache=False):
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        if _return_cache:
            raise NotImplementedError()
        return x


if __name__ == "__main__":
    from tqdm import tqdm
    from mycv.utils.torch_utils import num_params, load_partial, weights_replace, model_benchmark
    from mycv.datasets.imagenet import imagenet_val

    if True: # evaluate teacher
        model = VGG11Teacher()
        model = model.cuda()
        model.eval()
        results = imagenet_val(model, img_size=224, input_norm='imagenet',
                               batch_size=64, workers=4)
        print(results)
        exit()

    # model = VCMClassify(
    #     stage1='',
    #     stage2='cheng5',
    #     stage3='efb0',
    #     num_cls=1000
    # )
    # teacher_weights = torch.load(MYCV_DIR / 'weights/resnet50-19c8e357.pth')
    # teacher_weights = torch.load(MYCV_DIR / 'weights/res50_imgnet200.pt')
    # load_partial(model.stage3, teacher_weights, verbose=True)
    # exit()
    # from mycv.models.nlaic.nlaic import Dec
    # model = Dec()
    # from mycv.external.compressai.waseda import Cheng2020Anchor
    # model = Cheng2020Anchor(N=192, enable_bpp=False)
    # model = model.g_s

    # model = cResNet(192, num_classes=1000)
    # model = VGG11aa(num_classes=1000)
    # model = ResNet_aa_spc(192, num_classes=1000)
    # model = EfficientNet_('efficientnet-b0', 192)
    # model = effnet_b0_aa(192)
    # model = EffNet_b0_aa(192)
    model = ViTaa()
    # load_partial(model, teacher_weights, verbose=True)

    # y = model(torch.rand(1, 192, 14, 14), _return_cache=True); exit()
    model_benchmark(model, input_shape=(192, 14, 14), n_cpu=100, n_cuda=1000)
    exit()

    # model = Transfer(in_ch=128, name='resnet50')
    # print(num_params(model.m1))
    # print(num_params(model.layer3))
    # debug = 1

    # model = VCMClassify(stage1='NLAIC', stage2='NLAIC', stage3='ori_res50', num_cls=1000)
    # weights = rename_weights(MYCV_DIR / 'weights/nlaic/nlaic_ms64.pt', 'encoder.', '')
    # load_partial(model.stage1, weights)
    # weights = rename_weights(MYCV_DIR / 'weights/nlaic/nlaic_ms64.pt', 'decoder.', '')
    # load_partial(model.stage2, weights)
    # weights = rename_weights('weights/mini_dec_res50_guide_best.pt', 'stage3.', '')
    # load_partial(model.stage3, MYCV_DIR / 'weights/resnet50-19c8e357.pth')
    # weights = rename_weights('weights/nlaic_ms64__res50_v2_l2_0/best.pt', 'stage3.', '')
    # load_partial(model.stage3, weights)
    model = model.cuda()
    model.eval()
    from mycv.datasets.imagenet import imagenet_val
    results = imagenet_val(model, split='val',
                          img_size=224, batch_size=16, workers=4, input_norm=False)
    # from mycv.datasets.comp_imagenet import imagenet_val
    # results = imagenet_val(model.stage3, split='val_ms64',
    #                       img_size=14, batch_size=16, workers=0)
    # from mycv.datasets.imcoding import nic_evaluate
    # results = nic_evaluate(model, input_norm=False, verbose=True, dataset='kodak')
    # print(results)

    # x = torch.rand(128, 3, 224, 224).cuda()
    # print(torch.cuda.max_memory_allocated()/1e9)
    # with torch.no_grad():
    #     y = model(x)
    # print(torch.cuda.max_memory_allocated()/1e9)
    # print(torch.cuda.memory_allocated()/1e9)
