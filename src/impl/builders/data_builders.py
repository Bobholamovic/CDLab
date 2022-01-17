# Custom data builders

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import constants
from utils.data_utils.augmentations import *
from utils.data_utils.preprocessors import *
from core.misc import DATA, R
from core.data import (
    build_train_dataloader, build_eval_dataloader, get_common_train_configs, get_common_eval_configs
)
        

@DATA.register_func('SVCD_train_dataset')
def build_svcd_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            Identity()),
        ), Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_SVCD,
        sets=('real',)
    ))

    from data.svcd import SVCDDataset
    return build_train_dataloader(SVCDDataset, configs, C)


@DATA.register_func('SVCD_eval_dataset')
def build_svcd_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(
        None,    
        Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_SVCD,
        sets=('real',)
    ))

    from data.svcd import SVCDDataset
    return DataLoader(
        SVCDDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=False,
        num_workers=C['num_workers'],
        drop_last=False,
        pin_memory=C['device']!='cpu'
    )


@DATA.register_func('LEVIRCD_train_dataset')
def build_levircd_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            Identity()), Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_LEVIRCD,
    ))

    from data.levircd import LEVIRCDDataset
    return build_train_dataloader(LEVIRCDDataset, configs, C)


@DATA.register_func('LEVIRCD_eval_dataset')
def build_levircd_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_LEVIRCD,
    ))

    from data.levircd import LEVIRCDDataset
    return DataLoader(
        LEVIRCDDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=False,
        num_workers=C['num_workers'],
        drop_last=False,
        pin_memory=C['device']!='cpu'
    )


@DATA.register_func('WHU_train_dataset')
def build_whu_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            Identity()), Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_WHU,
    ))

    from data.whu import WHUDataset
    return build_train_dataloader(WHUDataset, configs, C)


@DATA.register_func('WHU_eval_dataset')
def build_whu_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_WHU,
    ))

    from data.whu import WHUDataset
    return DataLoader(
        WHUDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=False,
        num_workers=C['num_workers'],
        drop_last=False,
        pin_memory=C['device']!='cpu'
    )