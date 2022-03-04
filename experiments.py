import torch


def main():
    from mycv.datasets.imagenet import imagenet_val

    from models.irvine2022wacv import BottleneckResNetBackbone
    model = BottleneckResNetBackbone(zdim=64, bottleneck='vqa')

    wpath = 'runs/irvine-vqa_6-gelu.pt'
    model.load_state_dict(torch.load(wpath)['model'])

    model = model.cuda()
    model.eval()

    model.init_testing()
    results = imagenet_val(model, batch_size=64, workers=4)
    num, bpp, bpd = model.testing_stats
    results.update({'bppix': bpp, 'bpdim': bpd})
    print(results)

    debug = 1


if __name__ == '__main__':
    main()
