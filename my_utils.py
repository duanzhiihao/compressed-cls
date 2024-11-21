from pathlib import Path
from collections import OrderedDict, defaultdict
import os
import math
import random
import itertools
import numpy as np
import cv2
import torch
import torch.nn as nn


def disable_multithreads():
    """ Disable multi-processing in numpy and cv2
    """
    os.environ["OMP_NUM_THREADS"]      = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"]      = "1" # export MKL_NUM_THREADS=6
    os.environ["NUMEXPR_NUM_THREADS"]  = "1" # export NUMEXPR_NUM_THREADS=6
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def increment_dir(dir_root='runs/', name='exp'):
    """ Increament directory name. E.g., exp_1, exp_2, exp_3, ...

    Args:
        dir_root (str, optional): root directory. Defaults to 'runs/'.
        name (str, optional): dir prefix. Defaults to 'exp'.
    """
    assert isinstance(dir_root, (str, Path))
    dir_root = Path(dir_root)
    n = 0
    while (dir_root / f'{name}_{n}').is_dir():
        n += 1
    name = f'{name}_{n}'
    return name


def get_model_weights(weights):
    """ Get model weights from a path str or a dict

    Args:
        weights (str, dict-like): a file path, or a dict, or the weights itself
    """
    if isinstance(weights, (str, Path)):
        weights = torch.load(weights, map_location='cpu')
    if 'model' in weights:
        weights = weights['model']
    assert isinstance(weights, OrderedDict), 'model weights should be an OrderedDict'
    weights: OrderedDict
    return weights


def load_partial(model, weights):
    ''' Load weights that have the same name
    
    Args:
        model (torch.nn.Module): model
        weights (str or dict): weights
        verbose (bool, optional): deprecated.
    '''
    external_state = get_model_weights(weights)

    self_state = model.state_dict()
    new_dic = OrderedDict()
    for k,v in external_state.items():
        if k in self_state and self_state[k].shape == v.shape:
            new_dic[k] = v
        else:
            debug = 1
    model.load_state_dict(new_dic, strict=False)

    msg = (f'{type(model).__name__}: {len(self_state)} layers,'
           f'saved: {len(external_state)} layers,'
           f'overlap & loaded: {len(new_dic)} layers.')
    print(msg)


def reset_model_parameters(model: nn.Module, verbose=True):
    history = [set(), set()]
    for m in model.modules():
        mtype = type(m).__name__
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
            history[0].add(mtype)
        else:
            history[1].add(mtype)
    if verbose:
        print(f'{type(model)}: reset parameters of {history[0]}; no effect on {history[1]}')


def adjust_lr_threestep(optimizer, cur_epoch, base_lr, total_epoch):
    """ Sets the learning rate to the initial LR decayed by 10 every total/3 epochs

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        cur_epoch (int): current epoch
        base_lr (float): base learning rate
        total_epoch (int): total epoch
    """
    assert total_epoch >= 3
    period = math.ceil(total_epoch / 3)
    lr = base_lr * (0.1 ** (cur_epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


