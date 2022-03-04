import torch
from mycv.utils.torch_utils import flops_benchmark, speed_benchmark
from models.irvine2022wacv import BottleneckResNetLayerWithIGDN, BaselinePlus, BottleneckVQa


def main():
    # model = BottleneckResNetLayerWithIGDN(24, 256, _flops_mode=True)
    model = BaselinePlus(24, 256, _flops_mode=True)
    # model = BottleneckVQa(64, 256, _flops_mode=True)
    model.eval()

    shape = (3, 224, 224)
    flops_benchmark(model, input_shape=shape)
    device = torch.device('cpu')
    speed_benchmark(model, input_shapes=[shape], device=device, bs=1, N=4000)

    debug = 1


if __name__ == '__main__':
    main()
