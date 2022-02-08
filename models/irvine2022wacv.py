import torch
import torch.nn as nn

from compressai.layers.gdn import GDN1
from compressai.models.google import FactorizedPrior


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
    def forward(self, x):
        # if not self.training:
        #     encoded_obj = self.encode(x)
        #     self.analyze_compressed_object(encoded_obj)
        #     decoded_obj = self.decode(encoded_obj)
        #     return decoded_obj

        # # if fine-tuning after "update"
        # if self.updated:
        #     encoded_output = self.encoder(x)
        #     decoder_input =\
        #         self.entropy_bottleneck.dequantize(self.entropy_bottleneck.quantize(encoded_output, 'dequantize'))
        #     decoder_input = decoder_input.detach()
        #     return self.decoder(decoder_input)

        z = self.encoder(x)
        z_hat, z_probs = self.entropy_bottleneck(z)
        x_hat = self.decoder(z_hat)
        return x_hat, z_probs

    # def update(self, force=False):
    #     super().update(force=force)
    #     self.updated = True

    # def encode(self, x, **kwargs):
    #     latent = self.encoder(x)
    #     compressed_latent = self.entropy_bottleneck.compress(latent)
    #     return compressed_latent, latent.size()[2:]

    # def decode(self, compressed_obj, **kwargs):
    #     compressed_latent, latent_shape = compressed_obj
    #     latent_hat = self.entropy_bottleneck.decompress(compressed_latent, latent_shape)
    #     return self.decoder(latent_hat)


class BottleneckResNetBackbone(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.bottleneck_layer = BottleneckResNetLayerWithIGDN(24, 256)

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
        x, probs = self.bottleneck_layer(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        y = self.fc(x)

        self.cache = [None, x2, x3, x4]
        return y, probs

    # def update(self):
    #     self.bottleneck_layer.update()
    #     self.updated = True


def main():
    
    debug = 1


if __name__ == '__main__':
    main()
