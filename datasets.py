from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import re
import random
import torch
import torch.utils.data
import torchvision as tv
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
from timm.utils import AverageMeter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The root directory of all datasets
DATASETS_ROOT = Path('~/datasets').expanduser().resolve()

dataset_paths = {
    # ImageNet: http://www.image-net.org
    'imagenet':       DATASETS_ROOT / 'imagenet',
    'imagenet-train': DATASETS_ROOT / 'imagenet/train',
    'imagenet-val':   DATASETS_ROOT / 'imagenet/val',
}


def parse_config_str(config_str: str):
    """

    Args:
        config_str (str): [description]

    ### Examples:
        >>> input_1: 'rand-re0.25'
        >>> output_1: {'rand': True, 're': 0.25}

        >>> input_2: 'baseline'
        >>> output_2: {'baseline': True}
    """
    configs = dict()
    for kv_pair in config_str.split('-'):
        result = re.split(r'(\d.*)', kv_pair)
        if len(result) == 1:
            k = result[0]
            configs[k] = True
        else:
            assert len(result) == 3 and result[2] == ''
            k, v, _ = re.split(r'(\d.*)', kv_pair)
            configs[k] = float(v)
    return configs


def get_tv_interpolation(name='bilinear'):
    """ get interpolation for torchvision

    Args:
        name (str, optional): 'bilinear' or 'bicubic'
    """
    if name == 'bilinear':
        # if float(tv.__version__[:-2]) < 0.10:
        #     interp = Image.BILINEAR
        #     print(f'Warning: torchvision version: {tv.__version__} do not support InterpolationMode.',
        #         'Using PIL.Image int instead...')
        interp = tvt.InterpolationMode.BILINEAR
    elif name == 'bicubic':
        # if float(tv.__version__[:-2]) < 0.10:
        #     interp = Image.BICUBIC
        #     print(f'Warning: torchvision version: {tv.__version__} do not support InterpolationMode.',
        #         'Using PIL.Image int instead...')
        interp = tvt.InterpolationMode.BICUBIC
    else:
        raise NotImplementedError()
    return interp


class IdentityTransform():
    def __call__(self, x):
        return x


class RandomInterpolation():
    value = ('bilinear', 'bicubic')


class RandomResizedCropInterp(tvt.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=None)
        self.interpolation = RandomInterpolation()

    def forward(self, img):
        interp = random.choice(self.interpolation.value)
        interp = get_tv_interpolation(interp)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return tvf.resized_crop(img, i, j, h, w, self.size, interp)


def get_input_normalization(name: str):
    """ get tvt.Normalize according to the name

    Args:
        name (str): 'imagenet', 'inception'
    """
    if name == 'imagenet':
        norm = tvt.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif name == 'inception':
        norm = tvt.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    elif not name:
        norm = IdentityTransform()
    else:
        raise ValueError(f'Unknown input_norm: {name}')
    return norm


def get_transform(tf_str: str, img_size: int, input_norm):
    """ Get transform given a config string

    Args:
        tf_str (str): transform congif string
        img_size (int): input image size
        input_norm ([type]): input normalization type
    """
    aug_dict = parse_config_str(tf_str)

    if aug_dict.pop('finetune', False):
        transform = [
            tvt.RandomResizedCrop(img_size, scale=(0.64, 1)),
            tvt.RandomHorizontalFlip(),
        ]
    else:
        transform = [
            # tvt.RandomResizedCrop(img_size),
            RandomResizedCropInterp(img_size),
            tvt.RandomHorizontalFlip(),
        ]

    if aug_dict.pop('baseline', False):
        transform.append(tvt.ToTensor())

    if aug_dict.pop('rand', False):
        # from timm.data.transforms import RandomResizedCropAndInterpolation
        from timm.data.random_erasing import RandomErasing
        from timm.data.auto_augment import rand_augment_transform

        assert input_norm == 'imagenet', f'input_norm required to be imagenet but get {input_norm}'
        if aug_dict.pop('inc', False):
            ra_str = 'rand-m9-mstd0.5-inc1'
        else:
            ra_str = 'rand-m9-mstd0.5'
        reprob = aug_dict.pop('re', 0.5)
        transform = [
            # RandomResizedCropAndInterpolation(img_size, interpolation='random'),
            RandomResizedCropInterp(img_size),
            tvt.RandomHorizontalFlip(),
            rand_augment_transform(
                config_str=ra_str,
                hparams={'translate_const': int(img_size*0.45),
                         'img_mean': tuple([round(255 * x) for x in IMAGENET_MEAN])}
            ),
            tvt.ToTensor(),
            RandomErasing(reprob, mode='pixel', max_count=1, device='cpu')
        ]
        assert len(aug_dict) == 0

    if aug_dict.pop('trivial', False):
        from torchvision.transforms.autoaugment import TrivialAugmentWide
        transform.extend([
            TrivialAugmentWide(interpolation=get_tv_interpolation('bilinear')),
            tvt.PILToTensor(),
            tvt.ConvertImageDtype(torch.float)
        ])

    if input_norm:
        norm = get_input_normalization(input_norm)
        transform.append(norm)

    reprob = aug_dict.pop('re', 0)
    if reprob > 0:
        transform.append(tvt.RandomErasing(p=reprob))

    if len(aug_dict) != 0:
        raise ValueError(f'Unknown transform augmentation str: {tf_str}')

    transform = tvt.Compose(transform)
    return transform


def get_trainloader(root_dir, aug: str, img_size: int, input_norm: str,
                    batch_size: int, workers: int, distributed=False):
    """ get training data loader

    Args:
        root_dir ([type]): [description]
        aug (str): transform congif string
        img_size (int): input image size
        input_norm ([type]): input normalization type
        batch_size (int): [description]
        workers (int): [description]
    """
    transform = get_transform(aug, img_size, input_norm)

    trainset = tv.datasets.ImageFolder(root=root_dir, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset) if distributed else None
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(sampler is None), num_workers=workers,
        pin_memory=True, drop_last=False, sampler=sampler
    )
    return trainloader


def get_valloader(split='val',
        img_size=224, crop_ratio=0.875, interp='bilinear', input_norm='imagenet',
        batch_size=1, workers=0
    ):
    _original = dataset_paths['imagenet'] / split
    root_dir = _original

    transform = tvt.Compose([
        tvt.Resize(round(img_size/crop_ratio), interpolation=get_tv_interpolation(interp)),
        tvt.CenterCrop(img_size),
        tvt.ToTensor(),
        get_input_normalization(input_norm)
    ])

    dataset = tv.datasets.ImageFolder(root=root_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False
    )
    return dataloader


@torch.no_grad()
def imcls_evaluate(model: torch.nn.Module, testloader):
    """ Image classification evaluation with a testloader.

    Args:
        model (torch.nn.Module): pytorch model
        testloader (torch.utils.data.Dataloader): test dataloader
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    nC = len(testloader.dataset.classes)

    # tp1_sum, tp5_sum, total_num = 0, 0, 0
    stats_avg_meter = defaultdict(AverageMeter)
    print(f'Evaluating {type(model)}, device={device}, dtype={dtype}')
    print(f'batch_size={testloader.batch_size}, num_workers={testloader.num_workers}')
    pbar = tqdm(testloader)
    for imgs, labels in pbar:
        # sanity check
        imgs: torch.FloatTensor
        assert (imgs.dim() == 4)
        nB = imgs.shape[0]
        labels: torch.LongTensor
        assert (labels.shape == (nB,)) and (labels.dtype == torch.int64)
        assert 0 <= labels.min() and labels.max() <= nC-1

        # forward pass, get prediction
        # _debug(imgs)
        imgs = imgs.to(device=device, dtype=dtype)
        # if hasattr(model, 'self_evaluate'):
        #     labels = labels.to(device=device)
        #     stats = model.self_evaluate(imgs, labels) # workaround
        # else:
        forward_ = getattr(model, 'forward_cls', model.forward)
        p = forward_(imgs)
        # reduce to top5
        _, top5_idx = torch.topk(p, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(labels.view(nB, 1), top5_idx.cpu())
        stats = {
            'top1': correct5[:, 0].float().mean().item(),
            'top5': correct5.any(dim=1).float().mean().item()
        }

        for k, v in stats.items():
            stats_avg_meter[k].update(float(v), n=nB)

        # logging
        msg = ''.join([f'{k}={v.avg:.4g}, ' for k,v in stats_avg_meter.items()])
        pbar.set_description(msg)
    pbar.close()

    # compute total statistics and return
    _random_key = list(stats_avg_meter.keys())[0]
    assert stats_avg_meter[_random_key].count == len(testloader.dataset)
    results = {k: v.avg for k,v in stats_avg_meter.items()}
    return results
