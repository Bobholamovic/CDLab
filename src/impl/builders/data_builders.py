# Custom data builders

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import constants
from utils.data_utils.augmentations import *
from utils.data_utils.preprocessors import *
from utils.data_utils.preprocessors import Preprocess
from core.misc import DATA, R
from core.data import (
    build_train_dataloader, build_eval_dataloader, get_common_train_configs, get_common_eval_configs
)


@DATA.register_func('AC-Szada_train_dataset')
def build_ac_szada_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Crop(C['crop_size']),
            Choose(                
                HorizontalFlip(), VerticalFlip(), 
                Rotate('90'), Rotate('180'), Rotate('270'),
                Shift(),
            )
        ), Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_szada import AC_SzadaDataset
    if C['num_workers'] != 0:
        R['Logger'].warn("Will use num_workers=0.")
    return DataLoader(
        AC_SzadaDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=0,  # No need to use multiprocessing
        drop_last=True,
        pin_memory=C['device']!='cpu'
    )


@DATA.register_func('AC-Szada_eval_dataset')
def build_ac_szada_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_szada import AC_SzadaDataset
    return DataLoader(
        AC_SzadaDataset(**configs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False
    )


@DATA.register_func('AC-Tiszadob_train_dataset')
def build_ac_tiszadob_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Crop(C['crop_size']),
            Choose(                
                HorizontalFlip(), VerticalFlip(), 
                Rotate('90'), Rotate('180'), Rotate('270'),
                Shift(),
            )
        ), Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_tiszadob import AC_TiszadobDataset
    if C['num_workers'] != 0:
        R['Logger'].warn("Will use num_workers=0.")
    return DataLoader(
        AC_TiszadobDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=0,  # No need to use multiprocessing
        drop_last=True,
        pin_memory=C['device']!='cpu'
    )


@DATA.register_func('AC-Tiszadob_eval_dataset')
def build_ac_tiszadob_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_tiszadob import AC_TiszadobDataset
    return DataLoader(
        AC_TiszadobDataset(**configs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False
    )


@DATA.register_func('OSCD_train_dataset')
def build_oscd_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Crop(C['crop_size']),
            FlipRotate()
        ), Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_OSCD,
        cache_level=2,
    ))

    from data.oscd import OSCDDataset
    if C['num_workers'] != 0:
        R['Logger'].warn("Will use num_workers=0.")
    return DataLoader(
        OSCDDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=0,  # Disable multiprocessing
        drop_last=True,
        pin_memory=C['device']!='cpu'
    )


@DATA.register_func('OSCD_eval_dataset')
def build_oscd_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_OSCD,
        cache_level=2
    ))

    from data.oscd import OSCDDataset
    return DataLoader(
        OSCDDataset(**configs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False
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
    return build_eval_dataloader(SVCDDataset, configs)


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
    return build_eval_dataloader(LEVIRCDDataset, configs)


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
    return build_eval_dataloader(WHUDataset, configs)