import math
import torch
import torch.nn as nn

from compressai.layers.gdn import GDN1
from compressai.models.google import FactorizedPrior
from models.mobile import MobileCloudBase


class BottleneckResNetLayerWithIGDN(FactorizedPrior):
    def __init__(self, num_enc_channels=16, num_target_channels=256):
        super().__init__(N=num_enc_channels, M=num_enc_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 4),
            nn.Conv2d(num_enc_channels * 4, num_enc_channels * 2, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 2),
            nn.Conv2d(num_enc_channels * 2, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_enc_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )
        self.updated = False

    @torch.autocast('cuda', enabled=False)
    def forward_entropy(self, z):
        z = z.float()
        z_hat, z_probs = self.entropy_bottleneck(z)
        return z_hat, z_probs

    def forward(self, x):
        x = x.float()
        z = self.encoder(x)
        z_hat, z_probs = self.forward_entropy(z)
        x_hat = self.decoder(z_hat)
        return x_hat, z_probs


class BottleneckVQa(nn.Module):
    def __init__(self, num_enc_channels=64, num_target_channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels),
            nn.Conv2d(num_enc_channels, num_enc_channels, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels),
            nn.Conv2d(num_enc_channels, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_enc_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )
        self.updated = False
        from mycv.models.vqvae.myvqvae import MyCodebookEMA
        self.codebook = MyCodebookEMA(256, embedding_dim=num_enc_channels, commitment_cost=0.25)

    @torch.autocast('cuda', enabled=False)
    def forward_entropy(self, z):
        z = z.float()
        vq_loss, z_quantized = self.codebook.forward_train(z)
        nB, nC, nH, nW = z.shape
        z_probs = torch.ones(nB, nH, nW, device=z.device) / float(self.codebook._num_embeddings)
        # z_hat, z_probs = self.entropy_bottleneck(z)
        return vq_loss, z_quantized, z_probs

    def forward(self, x):
        x = x.float()
        z = self.encoder(x)
        vq_loss, z_hat, z_probs = self.forward_entropy(z)
        x_hat = self.decoder(z_hat)
        return x_hat, z_probs, vq_loss


class BottleneckResNetBackbone(MobileCloudBase):
    def __init__(self, num_classes=1000, zdim=24, bottleneck='factorized'):
        super().__init__()
        if bottleneck == 'factorized':
            self.bottleneck_layer = BottleneckResNetLayerWithIGDN(zdim, 256)
        elif bottleneck == 'vqa':
            self.bottleneck_layer = BottleneckVQa(zdim, 256)
        else:
            raise ValueError()

        from torchvision.models.resnet import resnet50
        # from timm.models.resnet import resnet50
        resnet_model = resnet50(pretrained=True, num_classes=num_classes)

        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc
        # self.inplanes = resnet_model.inplanes
        # self.updated = False
        self.cache = [None, None, None, None]

    def forward(self, x):
        x, probs, vq_loss = self.bottleneck_layer(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        y = self.fc(x)

        self.cache = [None, x2, x3, x4]
        if not self.training:
            return y, probs
        return y, probs, vq_loss

    def forward_entropy(self, z):
        raise NotImplementedError()

    # def update(self):
    #     self.bottleneck_layer.update()
    #     self.updated = True


def main():
    
    debug = 1


if __name__ == '__main__':
    main()
