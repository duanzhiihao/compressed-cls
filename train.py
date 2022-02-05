from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn.functional as tnf
import torch.cuda.amp as amp
# from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from mycv.paths import MYCV_DIR, IMAGENET_DIR
from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import load_partial, set_random_seeds, ModelEMA, reset_model_parameters
import  mycv.utils.lr_schedulers as lr_schedulers
from mycv.datasets.imcls import get_trainloader, imcls_evaluate
from mycv.datasets.imagenet import get_valloader
from mycv.datasets.constants import IMAGENET_MEAN, IMAGENET_STD
from models.ccls import VCMClassify


def train():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',    type=str,  default='vcm')
    parser.add_argument('--group',      type=str,  default='imagenet')
    parser.add_argument('--s1',         type=str,  default='nlaic_ms64')
    parser.add_argument('--s2',         type=str,  default='')
    parser.add_argument('--s3',         type=str,  default='aavgg16')
    parser.add_argument('--guide',      type=str,  default=False)
    parser.add_argument('--lambdas',    type=float,default=[1,1,1,1], nargs='+')
    parser.add_argument('--kdloss',     action='store_true')
    parser.add_argument('--lr',         type=float,default=0.01)
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--load',       type=str,  default='')
    parser.add_argument('--loadoptim',  action='store_true')
    parser.add_argument('--batch_size', type=int,  default=256)
    parser.add_argument('--equiv_bs',   type=int,  default=None)
    # parser.add_argument('--amp',        type=bool, default=False)
    # parser.add_argument('--amp',        action='store_true')
    parser.add_argument('--ema',        type=bool, default=False)
    parser.add_argument('--optimizer',  type=str,  default='sgd', choices=['adamw', 'sgd'])
    parser.add_argument('--epochs',     type=int,  default=40)
    parser.add_argument('--metric',     type=str,  default='top1')
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=4)
    parser.add_argument('--fixseed',    action='store_true')
    parser.add_argument('--nopretrain', action='store_true')
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    cfg = parser.parse_args()
    # model
    cfg.img_size = 224
    cfg.sync_bn = False
    # optimizer
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0001
    cfg.accum_batch_size = cfg.equiv_bs if cfg.equiv_bs is not None else cfg.batch_size
    cfg.accum_num = max(1, round(cfg.accum_batch_size // cfg.batch_size))
    # EMA
    cfg.ema_decay = 0.9999
    cfg.ema_warmup_epochs = 4

    # check arguments
    metric:     str = cfg.metric.lower()
    epochs:     int = cfg.epochs
    print(cfg, '\n')
    if cfg.fixseed: # fix random seeds for reproducibility
        set_random_seeds(1)
    torch.backends.cudnn.benchmark = True
    # device setting
    assert torch.cuda.is_available()
    for _id in cfg.device:
        print(f'Using device {_id}:', torch.cuda.get_device_properties(_id))
    device = torch.device(f'cuda:{cfg.device[0]}')
    bs_each = cfg.batch_size // len(cfg.device)
    print()
    print('Batch size on each single GPU =', bs_each)
    print(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
    print(f'Effective batch size: {cfg.accum_batch_size}', '\n')

    # Dataset
    print('Initializing Datasets and Dataloaders...')
    if cfg.group.startswith('imagenet'):
        train_split = 'train'
        val_split = 'val'
        cfg.num_class = 1000
        eval_interval = 1
    elif cfg.group == 'mini200':
        raise DeprecationWarning()
        train_split = 'train200_600'
        val_split = 'val200_600'
        cfg.num_class = 200
        assert 'res' in cfg.s3
        teacher_weights = torch.load(MYCV_DIR / 'weights/res50_imgnet200.pt')['model']
        eval_interval = 2
    else:
        raise NotImplementedError()
    # training set
    trainloader = get_trainloader(root_dir=IMAGENET_DIR/'train',
        aug='baseline', img_size=cfg.img_size, input_norm=False,
        batch_size=cfg.batch_size, workers=cfg.workers
    )
    # test set
    testloader = get_valloader(split='val',
        img_size=224, crop_ratio=1, interp='bicubic', input_norm=False,
        cache=True, batch_size=cfg.batch_size//4, workers=cfg.workers//2
    )

    # Initialize model
    if cfg.guide:
        if 'res' in cfg.s3:
            print('Using target model resnet-50...')
            from mycv.models.cls.resnet import resnet50
            tgtmodel = resnet50(num_classes=cfg.num_class)
            weights = torch.load(MYCV_DIR / 'weights/resnet/resnet50-19c8e357.pth')
            tgtmodel.load_state_dict(weights)
        elif 'b0' in cfg.s3:
            print('Using target model efficientnet-b0...')
            from mycv.external.timm.efficientnet import EfficientNetWrapper
            tgtmodel = EfficientNetWrapper()
            wpath = MYCV_DIR / 'weights/efficientnet/efb0_7.pt'
            load_partial(tgtmodel, wpath, verbose=True)
        elif 'vgg11' in cfg.s3:
            print('Using target model VGG11...')
            from mycv.models.cls.vgg import VGG11
            tgtmodel = VGG11(num_classes=cfg.num_class, pretrained=True)
        else:
            raise ValueError()

        tgtmodel = tgtmodel.to(device=device)
        # tgtmodel.eval()

        print(f'Using {cfg.guide} distance loss...')
        if cfg.guide == 'l1':
            dist_loss = torch.nn.L1Loss(reduction='mean')
        elif cfg.guide == 'l2':
            dist_loss = torch.nn.MSELoss(reduction='mean')
        elif cfg.guide == 'cos':
            dist_loss = torch.nn.CosineSimilarity(dim=1)
        elif cfg.guide == 'sl1':
            dist_loss = torch.nn.SmoothL1Loss(beta=0.1, reduction='mean')
        else:
            raise ValueError()
    else:
        tgtmodel = dist_loss = None

    input_mean = torch.FloatTensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device=device)
    input_std  = torch.FloatTensor(IMAGENET_STD).view(1, 3, 1, 1).to(device=device)

    model = VCMClassify(
        stage1=cfg.s1,
        stage2=cfg.s2,
        stage3=cfg.s3,
        num_cls=cfg.num_class
    )

    if cfg.nopretrain:
        reset_model_parameters(model.stage3, verbose=True)
    model = model.to(device)

    # different optimization setting for different layers
    parameters = get_parameters(model.stage3, cfg)
    print('Parameter groups:', [len(pg['params']) for pg in parameters])

    # optimizer
    if cfg.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr)
    else:
        raise NotImplementedError()
    del parameters
    # AMP
    scaler = amp.GradScaler(enabled=not cfg.s1.startswith('cheng'))

    log_parent = Path(f'runs/{cfg.project}')
    if cfg.resume:
        # resume
        run_name = cfg.resume
        log_dir = log_parent / run_name
        assert log_dir.is_dir()
        checkpoint = torch.load(log_dir / 'last.pt')
        # print(list(checkpoint.keys()))
        model.stage3.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_fitness = checkpoint.get(metric, 0)
        results = {metric: checkpoint.get(metric, 0)}
        wb_id = open(log_dir / 'wandb_id.txt', 'r').read().strip()
    else:
        # new experiment
        run_name = increment_dir(dir_root=log_parent,
                    name=f'{cfg.s1}_{cfg.s2}_{cfg.s3}_{cfg.guide}')
        log_dir = log_parent / run_name # wandb logging dir
        os.makedirs(log_dir, exist_ok=False)
        print(str(model), file=open(log_dir / 'model.txt', 'w'))
        best_fitness = 0
        results = {metric: 0}
        wb_id = None
        start_epoch = 0

        if cfg.load:
            print(f'\nUsing pre-trained weights from {cfg.load}...')
            checkpoint = torch.load(log_parent / cfg.load / 'last.pt', map_location=device)
            # model.stage3.load_state_dict(checkpoint['model'])
            load_partial(model.stage3, checkpoint)
            if cfg.loadoptim:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])

    # initialize wandb
    print()
    wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name,
                config=cfg, dir='runs/', mode=cfg.wbmode, resume='allow', id=wb_id)
    cfg = wbrun.config
    cfg.log_dir = log_dir
    cfg.wandb_id = wbrun.id
    with open(log_dir / 'wandb_id.txt', 'a') as f:
        f.write(wbrun.id + '\n')

    # Exponential moving average
    if cfg.ema:
        raise NotImplementedError() # accumulate
        ema = ModelEMA(model, decay=cfg.ema_decay)
        ema.updates = start_epoch * len(trainloader)  # set EMA updates
        ema.warmup = cfg.ema_warmup_epochs * len(trainloader) # 4 epochs
        # cfg.ema_start_updates = ema.updates
        # cfg.ema_warmup_iters = ema.warmup
    else:
        ema = None

    if len(cfg.device) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.device)
        if cfg.guide:
            tgtmodel = torch.nn.DataParallel(tgtmodel, device_ids=cfg.device)

    # ======================== start training ========================
    pbar_title = ('%-8s' * 14) % (
        'Epoch', 'GPU_mem', 'lr',
        'ce', 'x1', 'x2', 'x3', 'x4', 'kd', 'trs', 'loss', 'tr_acc', 'teach', metric
    )
    for epoch in range(start_epoch, epochs):
        if cfg.nopretrain:
            lr_schedulers.adjust_lr_threestep(optimizer, epoch, cfg.lr, epochs)
        else:
            adjust_learning_rate(optimizer, epoch, cfg.lr, epochs)
        model.train()
        optimizer.zero_grad()

        time.sleep(0.1)
        train_cls, train_trs, train_loss, train_acc = [0.0] * 4
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, (imgs, labels) in pbar:
            niter = epoch * len(trainloader) + i
            # debugging
            # if True:
            #     import matplotlib.pyplot as plt
            #     for im, lbl in zip(imgs, labels):
            #         im = im.permute(1,2,0).numpy()
            #         plt.imshow(im); plt.show()
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            nB, nC, nH, nW = imgs.shape

            # classification loss
            with amp.autocast(enabled=not cfg.s1.startswith('cheng')):
                if cfg.guide:
                    # student
                    p_logits, x1, x2, x3, x4 = model(imgs, _return_cache=True)
                    # normalize image
                    imgs = imgs.sub_(input_mean).div_(input_std)
                    # teacher
                    with torch.no_grad():
                        tgt_logits, t1, t2, t3, t4 = tgtmodel(imgs, _return_cache=True)
                        assert tgt_logits.shape == (nB, cfg.num_class)
                    tacc = cal_acc(tgt_logits.detach(), labels)
                else:
                    p_logits = model(imgs)
                    t1 = t2 = t3 = t4 = x1 = x2 = x3 = x4 = tgt_logits = None
                    tacc = -1

                assert p_logits.shape == (nB, cfg.num_class)
                l_cls = tnf.cross_entropy(p_logits, labels) #* nB

                if cfg.guide:
                    l_trs = []
                    for fake, real in zip([x1,x2,x3,x4], [t1,t2,t3,t4]):
                        if fake is None:
                            l_trs.append(torch.zeros(1, device=device))
                        else:
                            assert fake.shape == real.shape, f'{fake.shape}, {real.shape}'
                            # _lt = tnf.l1_loss(fake, real, reduction='mean')
                            _lt = dist_loss(fake, real)
                            if cfg.guide == 'cos':
                                _lt = _lt.mean()
                            l_trs.append(_lt)
                    if cfg.kdloss:
                        p_log = tnf.log_softmax(p_logits, dim=1)
                        tgt_log = tnf.log_softmax(tgt_logits, dim=1)
                        l_kd = tnf.kl_div(p_log, tgt_log, log_target=True)
                    else:
                        l_kd = torch.zeros(1, device=device)
                    l_x1, l_x2, l_x3, l_x4 = l_trs
                    lmb1, lmb2, lmb3, lmb4 = cfg.lambdas
                    l_trs = lmb1*l_x1 + lmb2*l_x2 + lmb3*l_x3 + lmb4*l_x4 + l_kd
                    # l_trs = l_trs #* nB
                else:
                    l_x1, l_x2, l_x3, l_x4, l_kd, l_trs = torch.zeros(6 , device=device)
                loss = l_cls + l_trs
                loss = loss / cfg.accum_num

            scaler.scale(loss).backward()
            if niter % cfg.accum_num == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # if cfg.guide:
            #     model.stage3.cache = []
            #     tgtmodel.cache = []

            if ema:
                ema.update(model)
            # Scheduler
            # scheduler.step()

            # logging
            cur_lr = optimizer.param_groups[0]['lr']
            acc = cal_acc(p_logits.detach(), labels)
            train_cls  = (train_cls*i + l_cls.item()) / (i+1)
            train_trs  = (train_trs*i + l_trs.item()) / (i+1)
            train_loss = (train_loss*i + loss.item()) / (i+1)
            train_acc  = (train_acc*i + acc) / (i+1)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            s = ('%-8s' * 2 + '%-8.4g' * 12) % (
                f'{epoch}/{epochs-1}', f'{mem:.3g}G', cur_lr,
                train_cls, l_x1.item(), l_x2.item(), l_x3.item(), l_x4.item(), l_kd.item(),
                train_trs, train_loss, 100*train_acc, 100*tacc, 100*results[metric]
            )
            pbar.set_description(s)
            # Weights & Biases logging
            if niter % 100 == 0:
                torch.cuda.reset_peak_memory_stats()
                wbrun.log({
                    'general/lr': cur_lr,
                    'metric/train_acc': train_acc,
                    'train/train_loss': train_loss,
                    'train/train_trs': train_trs,
                    'train/train_acc': train_acc,
                    'ema/n_updates': ema.updates if ema is not None else 0,
                    'ema/decay': ema.get_decay() if ema is not None else 0
                }, step=niter)
                # logging end
            del l_x1, l_x2, l_x3, l_x4, l_kd, l_trs, l_cls, loss
            del p_logits, tgt_logits, t1, t2, t3, t4
            del imgs, labels
            # ----Mini batch end
        # ----Epoch end

        # save last checkpoint
        _eval_model = getattr(model, 'module', model)
        _save_model = ema.ema if ema is not None else _eval_model
        checkpoint = {
            'model'    : _save_model.stage3.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler'   : scaler.state_dict(),
            'epoch'    : epoch
        }

        # Evaluation
        if epoch % eval_interval == 0:
            _log_dic = {'general/epoch': epoch}
            results = imcls_evaluate(_eval_model, testloader=testloader)
            _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})
            if ema is not None:
                results = imcls_evaluate(ema.ema, testloader=testloader)
                _log_dic.update({'metric/ema_val_'+k: v for k,v in results.items()})
            wbrun.log(_log_dic, step=niter)
            # Write evaluation results
            res = s + '||' + '%10.4g' * 1 % (results[metric])
            with open(log_dir / 'results.txt', 'a') as f:
                f.write(res + '\n')
            # record metric into checkpoint
            checkpoint[metric] = results[metric]
            # save best checkpoint
            if results[metric] > best_fitness:
                best_fitness = results[metric]
                torch.save(checkpoint, log_dir / 'best.pt')
        # save last checkpoint
        torch.save(checkpoint, log_dir / 'last.pt')
        del checkpoint, _save_model, _eval_model
        # ----Epoch end
        print(results)
    # ----Training end


def get_parameters(model, cfg):
    # different optimization setting for different layers
    pgb, pgw, pgo = [], [], []
    pg_info = {
        'bn/bias': [],
        'weights': [],
        'other': []
    }
    for k, v in model.named_parameters():
        assert isinstance(k, str) and isinstance(v, torch.Tensor)
        if ('.bn' in k) or ('.bias' in k): # batchnorm or bias
            pgb.append(v)
            pg_info['bn/bias'].append((k, v.shape))
        elif '.weight' in k: # conv or linear weights
            pgw.append(v)
            pg_info['weights'].append((k, v.shape))
        else: # other parameters
            pgo.append(v)
            pg_info['other'].append((k, v.shape))
    parameters = [
        {'params': pgb, 'lr': cfg.lr, 'weight_decay': 0.0},
        {'params': pgw, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
        {'params': pgo, 'lr': cfg.lr, 'weight_decay': 0.0}
    ]

    print('optimizer parameter groups [b,w,o]:', [len(pg['params']) for pg in parameters], '\n')
    print('pgo parameters:')
    for k, vshape in pg_info['other']:
        print(k, vshape)
    print()
    return parameters


def adjust_learning_rate(optimizer, epoch, base_lr, total_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    assert total_epoch % 2 == 0
    if epoch < round(total_epoch * 3/4) - 1:
        lrf = 1
    elif epoch != total_epoch - 1:
        lrf = 0.1
    else: # last epoch
        assert epoch == total_epoch - 1
        lrf = 0.01
    lr = base_lr * lrf
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cal_acc(p: torch.Tensor, labels: torch.Tensor):
    p, labels = p.detach().clone(), labels.detach().clone()
    assert not p.requires_grad and p.device == labels.device
    if p.dim() == 2:
        assert p.shape[0] == labels.shape[0]
        _, p = torch.max(p, dim=1)
    else:
        assert p.dim() == 1 and p.shape == labels.shape
        p = torch.sigmoid(p)
        p[p<0.5] = 0
        p[p>=0.5] = 1
    # assert p.dtype == labels.dtype
    tp = (p == labels)
    assert tp.dtype == torch.bool and tp.dim() == 1
    acc = tp.to(dtype=torch.float).mean().item()
    return acc


if __name__ == '__main__':
    print()
    train()
