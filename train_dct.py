from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.cuda.amp as amp

from mycv.paths import IMAGENET_DIR, MYCV_DIR
from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds, ModelEMA, load_partial
import mycv.utils.lr_schedulers as lr_schedulers
from mycv.utils.image import save_tensor_images

from datasets import jpeg_dct_trainloader, jpeg_dct_valloader, imcls_dct_evaluate


def main():
    print()
    trainw = TrainWrapper()
    trainw.main()


def get_config():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--project',    type=str,  default='imagenet')
    parser.add_argument('--group',      type=str,  default='jpeg24')
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    # model setting
    parser.add_argument('--model',      type=str,  default='res50_drfa')
    # resume setting
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--pretrain',   type=str,  default='')
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=256)
    parser.add_argument('--accum_bs',   type=int,  default=None)
    parser.add_argument('--optimizer',  type=str,  default='sgd')
    parser.add_argument('--lr',         type=float,default=0.1)
    parser.add_argument('--lr_sched',   type=str,  default='threestep')
    parser.add_argument('--wdecay',     type=float,default=0.0001)
    # training setting
    parser.add_argument('--epochs',     type=int,  default=24)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=False)
    # miscellaneous training setting
    parser.add_argument('--skip_eval0', action='store_true')
    # device setting
    parser.add_argument('--fixseed',    action='store_true')
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=8)
    cfg = parser.parse_args()

    # model
    cfg.img_size = 224
    cfg.input_norm = 'imagenet'
    # optimizer
    cfg.momentum = 0.9
    cfg.accum_batch_size = cfg.accum_bs or cfg.batch_size
    cfg.accum_num = max(1, round(cfg.accum_batch_size // cfg.batch_size))
    # EMA
    cfg.ema_warmup_epochs = round(cfg.epochs / 20)
    # logging
    cfg.metric = 'top1' # metric to save best model

    if cfg.fixseed: # fix random seeds for reproducibility
        set_random_seeds(1)
    torch.backends.cudnn.benchmark = True
    return cfg


class TrainWrapper():
    def __init__(self) -> None:
        pass

    def set_device_(self):
        cfg = self.cfg

        # device setting
        assert torch.cuda.is_available()
        device = torch.device(f'cuda:{cfg.device[0]}')
        bs_each = cfg.batch_size // len(cfg.device)
        if True:
            for _id in cfg.device:
                print(f'Using device {_id}:', torch.cuda.get_device_properties(_id))
            print('Batch size on each single GPU =', bs_each)
            print(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
            print(f'Effective batch size: {cfg.accum_batch_size}', '\n')

        self.device = device
        self.cfg.bs_each = bs_each

    def set_dataset_(self):
        cfg = self.cfg

        if True:
            print('Initializing Datasets and Dataloaders...')
        if cfg.group == 'original':
            train_split = 'train'
            val_split = 'val224l_jpeg100'
        elif cfg.group[:4] == 'jpeg':
            train_split = f'train_{cfg.group}'
            val_split = f'val224l_{cfg.group}'
        elif cfg.group == 'mini200':
            train_split = 'train200_600'
            val_split = 'val200_600'
        else:
            train_split = f'train_{cfg.group}'
            val_split = f'val_{cfg.group}'

        # training set
        trainloader = jpeg_dct_trainloader(root_dir=IMAGENET_DIR/train_split,
            img_size=cfg.img_size, batch_size=cfg.batch_size, workers=cfg.workers
        )
        num_classes = len(trainloader.dataset.classes)
        # test set
        testloader = jpeg_dct_valloader(root_dir=IMAGENET_DIR/val_split,
            batch_size=cfg.bs_each//2, workers=cfg.workers//2
        )
        if True:
            print(f'Training root: {trainloader.dataset.root}')
            print(f'Val root: {testloader.dataset.root}')
            print(f'Number of classes = {num_classes}')
            print(f'First training data: {trainloader.dataset.samples[0]}')
            print(f'First val data: {testloader.dataset.samples[0]}', '\n')

        self.trainloader = trainloader
        self.testloader  = testloader
        self.cfg.num_classes = num_classes

    def set_model_(self):
        cfg = self.cfg
        mname, num_classes = cfg.model, cfg.num_classes

        if mname == 'res50_drfa':
            from models.dct import ResNetDRFA
            model = ResNetDRFA(num_classes=num_classes)
            wpath = MYCV_DIR / 'weights/resnet/resnet50-19c8e357.pth'
            load_partial(model, wpath, verbose=True)
        else:
            raise ValueError()
        if True:
            print(f'Using model {type(model)}, {num_classes} classes', '\n')

        if cfg.pretrain: # (partially or fully) initialize from pretrained weights
            load_partial(model, cfg.pretrain, verbose=True)

        self.model = model.to(self.device)
        if len(cfg.device) > 1: # DP mode
            print(f'DP mode on GPUs {cfg.device}', '\n')
            self.model = torch.nn.DataParallel(model, device_ids=cfg.device)

    def set_loss_(self):
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    def set_optimizer_(self):
        cfg, model = self.cfg, self.model

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
            {'params': pgw, 'lr': cfg.lr, 'weight_decay': cfg.wdecay},
            {'params': pgo, 'lr': cfg.lr, 'weight_decay': 0.0}
        ]

        # optimizer
        if cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum)
        elif cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optimizer}')

        if True:
            print('optimizer parameter groups:', [len(pg['params']) for pg in parameters], '\n')
            print('pgo parameters:')
            for k, vshape in pg_info['other']:
                print(k, vshape)
            print()

        self.optimizer = optimizer
        self.scaler = amp.GradScaler(enabled=cfg.amp) # Automatic mixed precision

    def set_logging_dir_(self):
        cfg = self.cfg

        log_parent = Path(f'runs/{cfg.project}')
        if cfg.resume: # resume
            assert not cfg.pretrain, '--resume not compatible with --pretrain'
            run_name = cfg.resume
            log_dir = log_parent / run_name
            assert log_dir.is_dir(), f'Try to resume from {log_dir} but it does not exist'
            ckpt_path = log_dir / 'last.pt'
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch']
            results = checkpoint.get('results', defaultdict(float))
            if True:
                print(f'Resuming run {log_dir}. Loaded checkpoint from {ckpt_path}.',
                      f'Epoch={start_epoch}, results={results}')
        else: # new experiment
            _base = f'{cfg.model}'
            run_name = increment_dir(dir_root=log_parent, name=_base)
            log_dir = log_parent / run_name # logging dir
            if True:
                os.makedirs(log_dir, exist_ok=False)
                print(str(self.model), file=open(log_dir / 'model.txt', 'w'))
                json.dump(cfg.__dict__, fp=open(log_dir / 'config.json', 'w'), indent=2)
                print('Training config:\n', cfg, '\n')
            start_epoch = 0
            results = defaultdict(float)

        cfg.log_dir = str(log_dir)
        self._log_dir      = log_dir
        self._start_epoch  = start_epoch
        self._results      = results
        self._best_fitness = results[cfg.metric]

    def set_wandb_(self):
        cfg = self.cfg

        # check if there is a previous run to resume
        wbid_path = self._log_dir / 'wandb_id.txt'
        if os.path.exists(wbid_path):
            run_ids = open(wbid_path, mode='r').read().strip().split('\n')
            rid = run_ids[-1]
        else:
            rid = None
        # initialize wandb
        import wandb
        run_name = self._log_dir.stem
        wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name,
                           config=cfg, dir='runs/', id=rid, resume='allow',
                           save_code=True, mode=cfg.wbmode)
        cfg = wbrun.config
        cfg.wandb_id = wbrun.id
        with open(wbid_path, mode='a') as f:
            print(wbrun.id, file=f)

        self.wbrun = wbrun
        self.cfg = cfg

    def set_ema_(self):
        cfg = self.cfg

        # Exponential moving average
        if cfg.ema:
            warmup = cfg.ema_warmup_epochs * len(self.trainloader)
            decay = 0.9999
            ema = ModelEMA(self.model, decay=decay, warmup=warmup)
            start_iter = self._start_epoch * len(self.trainloader)
            ema.updates = start_iter // cfg.accum_num # set EMA update number

            if cfg.resume:
                ckpt_path = self._log_dir / 'last_ema.pt'
                assert ckpt_path.is_file(), f'Cannot find EMA checkpoint: {ckpt_path}'
                ema.ema.load_state_dict(torch.load(ckpt_path)['model'])

            if True:
                print(f'Using EMA with warmup_epochs={cfg.ema_warmup_epochs},',
                      f'decay={decay}, past_updates={ema.updates}\n',
                      f'Loaded EMA from {ckpt_path}\n' if cfg.resume else '')
        else:
            ema = None

        self.ema = ema

    def main(self):
        # config
        self.cfg = get_config()

        # core
        self.set_device_()
        self.set_dataset_()
        self.set_model_()
        self.set_loss_()
        self.set_optimizer_()

        # logging
        self.set_logging_dir_()
        if True:
            self.set_wandb_()

        self.set_ema_()

        cfg = self.cfg
        model = self.model

        # ======================== start training ========================
        for epoch in range(self._start_epoch, cfg.epochs):
            time.sleep(0.1)

            pbar = enumerate(self.trainloader)
            if True:
                self.init_logging_()
                if not (epoch == self._start_epoch and cfg.skip_eval0):
                    # if cfg.skip_eval0, then skip the first evaluation
                    self.evaluate(epoch, niter=epoch*len(self.trainloader))

                print('\n' + self._pbar_title)
                pbar = tqdm(pbar, total=len(self.trainloader))

            self.adjust_lr_(epoch)
            model.train()
            for bi, (x_y, x_cb, x_cr, labels) in pbar:
                niter = epoch * len(self.trainloader) + bi

                x_y = x_y.to(device=self.device)
                x_cb = x_cb.to(device=self.device)
                x_cr = x_cr.to(device=self.device)
                labels = labels.to(device=self.device)

                # forward
                with amp.autocast(enabled=cfg.amp):
                    p = model(x_y, x_cb, x_cr)
                    assert p.shape == (x_y.shape[0], cfg.num_classes)
                    loss = self.loss_func(p, labels)
                    loss = loss / cfg.accum_num
                # loss is averaged over batch and gpus
                self.scaler.scale(loss).backward() # backward, update
                if niter % cfg.accum_num == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if cfg.ema:
                        self.ema.update(model)

                if True:
                    self.logging(pbar, epoch, bi, niter, labels, p, loss)
        if True:
            self.evaluate(epoch, niter)
        print('Training finished. results:', self._results)

    def adjust_lr_(self, epoch):
        cfg = self.cfg

        if cfg.lr_sched == 'threestep':
            lrf = lr_schedulers.threestep(epoch, cfg.epochs)
        elif cfg.lr_sched == 'cosine':
            lrf = lr_schedulers.cosine_lr(epoch, 1e-4, cfg.epochs)
        else:
            raise NotImplementedError()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cfg.lr * lrf

    def init_logging_(self):
        self._epoch_stat_keys = ['tr_loss', 'tr_acc']
        self._epoch_stat_vals = torch.zeros(len(self._epoch_stat_keys))
        sn = 5 + len(self._epoch_stat_keys)
        self._pbar_title = ('%-10s' * sn) % (
            'Epoch', 'GPU_mem', 'lr', *self._epoch_stat_keys, 'top1', 'top5',
        )
        self._msg = ''

    def logging(self, pbar, epoch, bi, niter, labels, p, loss):
        cfg = self.cfg
        cur_lr = self.optimizer.param_groups[0]['lr']
        acc = compute_acc(p.detach(), labels)
        stats = torch.Tensor([loss.item(), acc])
        self._epoch_stat_vals.mul_(bi).add_(stats).div_(bi+1)
        mem = torch.cuda.max_memory_allocated(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats()
        sn = len(self._epoch_stat_keys) + 2
        msg = ('%-10s' * 2 + '%-10.4g' + '%-10.4g' * sn) % (
            f'{epoch}/{cfg.epochs-1}', f'{mem:.3g}G', cur_lr,
            *self._epoch_stat_vals,
            100*self._results['top1'], 100*self._results['top5'],
        )
        pbar.set_description(msg)
        self._msg = msg

        # Weights & Biases logging
        if niter % 100 == 0:
            # save_tensor_images(imgs[:4], save_path=self._log_dir / 'imgs.png',
            #                    how_normed=cfg.input_norm)
            _log_dic = {
                'general/lr': cur_lr,
                'ema/n_updates': self.ema.updates if cfg.ema else 0,
                'ema0/decay': self.ema.get_decay() if cfg.ema else 0
            }
            _log_dic.update(
                {'train/'+k: v for k,v in zip(self._epoch_stat_keys, self._epoch_stat_vals)}
            )
            self.wbrun.log(_log_dic, step=niter)

    def evaluate(self, epoch, niter):
        # Evaluation
        _log_dic = {'general/epoch': epoch}
        results = imcls_dct_evaluate(self.model, testloader=self.testloader)
        _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})
        # save last checkpoint
        checkpoint = {
            'model'     : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scaler'    : self.scaler.state_dict(),
            'epoch'     : epoch,
            'results'   : results,
        }
        torch.save(checkpoint, self._log_dir / 'last.pt')
        self._save_if_best(checkpoint)

        if self.cfg.ema:
            results = imcls_dct_evaluate(self.ema.ema, testloader=self.testloader)
            _log_dic.update({f'metric/ema_val_'+k: v for k,v in results.items()})
            # save last checkpoint of EMA
            checkpoint = {
                'model'     : self.ema.ema.state_dict(),
                'epoch'     : epoch,
                'results'   : results,
            }
            torch.save(checkpoint, self._log_dir / 'last_ema.pt')
            self._save_if_best(checkpoint)

        # wandb log
        self.wbrun.log(_log_dic, step=niter)
        # Log evaluation results to file
        _cur_fitness = results[self.cfg.metric]
        msg = self._msg + '||' + '%10.4g' * 1 % (_cur_fitness)
        with open(self._log_dir / 'results.txt', 'a') as f:
            f.write(msg + '\n')

        self._results = results

    def _save_if_best(self, checkpoint):
        # save checkpoint if it is the best so far
        metric = self.cfg.metric
        fitness = checkpoint['results'][metric]
        if fitness > self._best_fitness:
            self._best_fitness = fitness
            svpath = self._log_dir / 'best.pt'
            torch.save(checkpoint, svpath)
            if True:
                print(f'Get highest {metric} = {fitness}. Saved to {svpath}.')


def compute_acc(p: torch.Tensor, labels: torch.LongTensor):
    assert not p.requires_grad and p.device == labels.device
    assert p.dim() == 2 and p.shape[0] == labels.shape[0]
    _, p_cls = torch.max(p, dim=1)
    tp = (p_cls == labels)
    acc = float(tp.sum()) / len(tp)
    assert 0 <= acc <= 1
    return acc * 100.0


if __name__ == '__main__':
    main()

    # from mycv.models.cls.resnet import resnet50
    # model = resnet50(num_classes=1000)
    # weights = torch.load('weights/resnet50-19c8e357.pth')
    # model.load_state_dict(weights)
    # model = model.cuda()
    # model.eval()
    # results = imagenet_val(model, img_size=224, batch_size=64, workers=4)
    # print(results['top1'])
