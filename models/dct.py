import torch
import torch.nn as nn
import torch.nn.functional as tnf

from torchvision.models.resnet import conv1x1, Bottleneck


class ResNetDRFA(nn.Module):
    '''
    Deconvolution-RFA model in
    ```
    Gueguen, L., Sergeev, A., Kadlec, B., Liu, R. and Yosinski, J.,2018.
    Faster neural networks straight from jpeg.
    Advances in Neural Information Processing Systems, 31, pp.3933-3944.
    ```
    '''
    def __init__(self, layers=[3,4,6,3], num_classes=1000):
        super().__init__()
        self.upcb = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upcr = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.inplanes = 192
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        block = Bottleneck
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
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

    def forward(self, x_y, x_cb, x_cr, _return_cache=False):
        x_cb = self.upcb(x_cb)
        x_cr = self.upcr(x_cr)
        x = torch.cat([x_y, x_cb, x_cr], dim=1)
        x = self.bn1(x)

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


if __name__ == '__main__':
    from mycv.utils.torch_utils import num_params, load_partial, flops_benchmark, speed_benchmark
    # y, cb, cr = [torch.load(f'{s}.pt') for s in ('y', 'cb', 'cr')]
    # y, cb, cr = [torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0) for x in (y, cb, cr)]

    model = ResNetDRFA()
    # y = model(y, cb, cr)

    inputs = (
        torch.randn(1, 64, 28, 28),
        torch.randn(1, 64, 14, 14),
        torch.randn(1, 64, 14, 14),
    )
    flops_benchmark(model, inputs=inputs)
    input_shapes = [(64, 28, 28), (64, 14, 14), (64, 14, 14)]
    speed_benchmark(model, input_shapes=input_shapes,
                    device=torch.device('cpu'), bs=1, N=500)
    speed_benchmark(model, input_shapes=input_shapes,
                    device=torch.device('cuda:0'), bs=1, N=10000)
    speed_benchmark(model, input_shapes=input_shapes,
                    device=torch.device('cuda:0'), bs=64, N=10000)

    debug = 1
