import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torchvision as tv

from compressai.entropy_models import EntropyBottleneck


class SimpleQuantization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if self.training:
            # x = x + torch.rand_like(x) - 0.5
            xd = x.detach()
            x = x + (torch.round(xd) - xd)
        else:
            assert not x.requires_grad
            x = torch.round_(x)
        return x, None


class ResNet50MC(nn.Module):
    def __init__(self, cut_after='layer1', entropy_model='bottleneck'):
        super().__init__()
        model = tv.models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

        # 3, 4, 6, 3
        stride2channel = {'layer1': 256, 'layer2': 512}
        channels = stride2channel[cut_after]
        if entropy_model == 'bottleneck':
            self.entropy_model = EntropyBottleneck(channels)
        if entropy_model == 'simple':
            self.entropy_model = SimpleQuantization()
        else:
            raise ValueError()
        self.cut_after = cut_after

    @amp.autocast(dtype=torch.float32)
    def forward_entropy(self, z):
        z, p_z = self.entropy_model(z)
        return z, p_z

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.cut_after == 'layer1':
            x, p_z = self.forward_entropy(x)
        x = self.layer2(x)
        if self.cut_after == 'layer2':
            x, p_z = self.forward_entropy(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        yhat = self.fc(x)

        return yhat, p_z

    def forward_cls(self, x):
        yhat, p_z = self.forward(x)
        return yhat


def main():
    from mycv.datasets.imagenet import imagenet_val
    model = ResNet50MC('layer2')
    # msd = torch.load('runs/best.pt')
    # model.load_state_dict(msd['model'])
    model = model.cuda()
    model.eval()

    resuls = imagenet_val(model, batch_size=64, workers=4)
    print(resuls)


if __name__ == '__main__':
    main()
