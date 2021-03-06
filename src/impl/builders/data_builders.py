# Custom data builders

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import constants
from utils.data_utils.augmentations import *
from utils.data_utils.preprocessors import *
from core.misc import DATA, R
from core.data import (
    build_train_dataloader, build_eval_dataloader, get_common_train_configs, get_common_eval_configs
)


@DATA.register_func('AC_Szada_train_dataset')
def build_AC_Szada_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Crop(C['crop_size']),
            Choose(                
                HorizontalFlip(), VerticalFlip(), 
                Rotate('90'), Rotate('180'), Rotate('270'),
                Shift(),
            )
        ), Normalize(0.0, 255.0), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_szada import AC_SzadaDataset
    if C['num_workers'] != 0:
        R['Logger'].warn("Will use num_workers=0.")
    return data.DataLoader(
        AC_SzadaDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=0,  # No need to use multiprocessing
        pin_memory=C['device']!='cpu',
        drop_last=True
    )


@DATA.register_func('AC_Szada_eval_dataset')
def build_AC_Szada_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(0.0, 255.0), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_szada import AC_SzadaDataset
    return data.DataLoader(
        AC_SzadaDataset(**configs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )


@DATA.register_func('AC_Tiszadob_train_dataset')
def build_AC_Tiszadob_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Crop(C['crop_size']),
            Choose(                
                HorizontalFlip(), VerticalFlip(), 
                Rotate('90'), Rotate('180'), Rotate('270'),
                Shift(),
            )
        ), Normalize(0.0, 255.0), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_tiszadob import AC_TiszadobDataset
    if C['num_workers'] != 0:
        R['Logger'].warn("Will use num_workers=0.")
    return data.DataLoader(
        AC_TiszadobDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=0,  # No need to use multiprocessing
        pin_memory=C['device']!='cpu',
        drop_last=True
    )


@DATA.register_func('AC_Tiszadob_eval_dataset')
def build_AC_Tiszadob_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(0.0, 255.0), None),
        root=constants.IMDB_AIRCHANGE
    ))

    from data.ac_tiszadob import AC_TiszadobDataset
    return data.DataLoader(
        AC_TiszadobDataset(**configs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )


@DATA.register_func('OSCD_train_dataset')
def build_OSCD_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Crop(C['crop_size']),
            FlipRotate()
        ), Normalize(zscore=True), None),
        root=constants.IMDB_OSCD,
        cache_level=2,
    ))

    from data.oscd import OSCDDataset
    if C['num_workers'] != 0:
        R['Logger'].warn("Will use num_workers=0.")
    return data.DataLoader(
        OSCDDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=0,  # Disable multiprocessing
        pin_memory=C['device']!='cpu',
        drop_last=True
    )


@DATA.register_func('OSCD_eval_dataset')
def build_OSCD_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(zscore=True), None),
        root=constants.IMDB_OSCD,
        cache_level=2
    ))

    from data.oscd import OSCDDataset
    return data.DataLoader(
        OSCDDataset(**configs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )


@DATA.register_func('Lebedev_train_dataset')
def build_Lebedev_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Scale([0.8, 3.0]),
            Shift(),),
            Crop(C['crop_size'])
        ), Normalize(0.0, 255.0), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_train_dataloader(LebedevDataset, configs, C)


@DATA.register_func('Lebedev_eval_dataset')
def build_Lebedev_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(0.0, 255.0), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_eval_dataloader(LebedevDataset, configs)