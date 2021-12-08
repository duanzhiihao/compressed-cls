import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torchvision as tv

from compressai.entropy_models import EntropyBottleneck


class IntegerQuantization(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.momentum = 0.99
        self.register_buffer('estimated_p', torch.ones(n_ch, 256).div_(256))
        self.estimated_p: torch.Tensor

        self.testing_stats = (0, 0.0, 0.0) # num, bpp, bpdim

    def init_testing(self):
        self.testing_stats = (0, 0.0, 0.0) # num, bpp, bpdim

    def _update_stats(self, x: torch.Tensor):
        assert not x.requires_grad
        x.clamp_(min=0, max=255).round_()
        # x = x.to(dtype=torch.int64)
        # x: (nB, nC, nH, nW)
        x = x.permute(1, 0, 2, 3).flatten(1)
        for i, x_i in enumerate(x):
            hist = torch.histc(x_i, bins=256, min=0, max=255)
            # hist = torch.bincount(x_i, minlength=256)
            assert hist.sum() == x.shape[1]
            pi = hist.float() / x.shape[1]
            self.estimated_p[i, :].mul_(self.momentum).add_(pi, alpha=1-self.momentum)
            assert torch.isclose(self.estimated_p[i, :].sum(), torch.ones(1, device=x.device))
        debug = 1

    def compute_likelihood(self, x: torch.Tensor):
        xhat = x.detach().clone().round_().to(torch.int64)
        assert xhat.min() >= 0
        p_x = []
        for i in range(xhat.shape[1]):
            indexs = xhat[:, i, :, :]
            p = self.estimated_p[i, indexs]
            p_x.append(p.unsqueeze(1))
        p_x = torch.cat(p_x, dim=1)
        return p_x

    def forward(self, x: torch.Tensor):
        if self.training:
            # x = x + torch.rand_like(x) - 0.5
            x = torch.clamp(x, max=255)
            xd = x.detach()
            x = x + (torch.round(xd) - xd)
            self._update_stats(xd)
        else:
            assert not x.requires_grad
            if x.max() > 255:
                print(f'Warning: x.max() = {x.max().item()} > 255')
            x = torch.round_(x)
        p_x = self.compute_likelihood(x)
        # if not self.training: # update testing stats
        #     num, bpp, bpdim = self.testing_stats
        #     bpp = 
        return x, p_x


def get_entropy_model(name, channels):
    if name == 'bottleneck':
        entropy_model = EntropyBottleneck(channels)
    elif name == 'quantize':
        entropy_model = IntegerQuantization(channels)
    else:
        raise ValueError()
    return entropy_model


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
        self.entropy_model = get_entropy_model(entropy_model, channels)
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


class VGG11MC(nn.Module):
    def __init__(self, cut_after=10, entropy_model='bottleneck'):
        super().__init__()
        model = tv.models.vgg.vgg11(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        self.cut_after = int(cut_after)
        channels = {5: 128, 10: 256, 15: 512, 20: 512}
        self.entropy_model = get_entropy_model(entropy_model, channels[self.cut_after])

    @amp.autocast(dtype=torch.float32)
    def forward_entropy(self, z):
        z, p_z = self.entropy_model(z)
        return z, p_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mi, module in enumerate(self.features):
            x = module(x)
            if mi == self.cut_after:
                x, p_z = self.forward_entropy(x)
        # x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, p_z

    def forward_cls(self, x):
        yhat, p_z = self.forward(x)
        return yhat


def main():
    from mycv.datasets.imagenet import imagenet_val
    model = ResNet50MC('layer2', 'quantize')
    # model = VGG11MC(10)
    # msd = torch.load('runs/best.pt')
    # model.load_state_dict(msd['model'])
    model = model.cuda()
    model.eval()

    resuls = imagenet_val(model, batch_size=4, workers=0)
    print(resuls)


if __name__ == '__main__':
    main()
