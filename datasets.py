from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision as tv

from jpeg2dct.numpy import load, loads


class DCTtrain(tv.datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            x_y, x_cb, x_cr = self.transform(sample)
        assert self.target_transform is None

        return x_y, x_cb, x_cr, target


class DCTtest(tv.datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]

        assert self.transform is None
        x_y, x_cb, x_cr = load(path)
        x_y, x_cb, x_cr = [torch.from_numpy(f).float().permute(2,0,1) \
                           for f in [x_y, x_cb, x_cr]]
        assert self.target_transform is None

        return x_y, x_cb, x_cr, target


class ToJPEGbits():
    def __init__(self) -> None:
        pass

    def __call__(self, img):
        assert isinstance(img, Image.Image)
        im = np.array(img)[:,:,::-1] # BGR image
        flag, bits = cv2.imencode('.jpg', im, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
        assert flag
        bits = bits.tobytes()
        return bits


class DecodeDCT():
    def __init__(self) -> None:
        pass

    def __call__(self, bits):
        y, cb, cr = loads(bits)
        y, cb, cr = [torch.from_numpy(f).float().permute(2,0,1) for f in [y, cb, cr]]
        return [y, cb, cr]


def jpeg_dct_trainloader(root_dir, img_size: int, batch_size: int, workers: int):
    transform = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(img_size),
        tv.transforms.RandomHorizontalFlip(),
        ToJPEGbits(),
        DecodeDCT()
    ])

    trainset = DCTtrain(root=root_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=False
    )
    return trainloader


def jpeg_dct_valloader(root_dir, batch_size: int, workers: int):
    trainset = DCTtest(root=root_dir)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=False
    )
    return trainloader



@torch.no_grad()
def imcls_dct_evaluate(model: torch.nn.Module, testloader):
    model.eval()
    device = next(model.parameters()).device
    forward_ = getattr(model, 'forward_cls', model.forward)
    nC = len(testloader.dataset.classes)

    tp1_sum, tp5_sum, total_num = 0, 0, 0
    pbar = tqdm(testloader)
    for x_y, x_cb, x_cr, labels in pbar:
        # sanity check
        x_y: torch.FloatTensor
        assert (x_y.dim() == 4), f'x_y shape {x_y.shape}'
        nB = x_y.shape[0]
        labels: torch.LongTensor
        assert (labels.shape == (nB,)) and (labels.dtype == torch.int64)
        assert 0 <= labels.min() and labels.max() <= nC-1

        # forward pass, get prediction
        x_y, x_cb, x_cr = x_y.to(device=device), x_cb.to(device=device), x_cr.to(device=device)
        p = forward_(x_y, x_cb, x_cr)
        # reduce to top5
        _, top5_idx = torch.topk(p, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(labels.view(nB, 1), top5_idx.cpu())
        _tp1 = correct5[:, 0].sum().item()
        _tp5 = correct5.any(dim=1).sum().item()
        tp1_sum += _tp1
        tp5_sum += _tp5

        # logging
        total_num += nB
        msg = f'top1: {tp1_sum/total_num:.4g}, top5: {tp5_sum/total_num:.4g}'
        pbar.set_description(msg)

    # compute total statistics and return
    assert total_num == len(testloader.dataset)
    acc_top1 = tp1_sum / total_num
    acc_top5 = tp5_sum / total_num
    results = {'top1': acc_top1, 'top5': acc_top5}
    return results


if __name__ == '__main__':
    from mycv.paths import IMAGENET_DIR
    from models.dct import ResNetDRFA
    model = ResNetDRFA()
    model = model.cuda()
    model.eval()

    # valloader = jpeg_dct_valloader(IMAGENET_DIR/'val224l_jpeg24', batch_size=64, workers=4)
    # results = imcls_dct_evaluate(model, valloader)
    # print(results)

    from tqdm import tqdm
    trainloader = jpeg_dct_trainloader(IMAGENET_DIR/'train_jpeg24',
        img_size=224, batch_size=256, workers=16)
    for x_y, x_cb, x_cr, y in tqdm(trainloader):
        debug = 1
