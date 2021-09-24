# Custom data builders

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

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


class _Identity:
    def __call__(self, *args):
        return args if len(args)>0 else args[0]
        

@DATA.register_func('Lebedev_train_dataset')
def build_Lebedev_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            _Identity()),
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
        transforms=(
        None,    
        Normalize(0.0, 255.0), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_eval_dataloader(LebedevDataset, configs)


@DATA.register_func('Lebedev_P2V_train_dataset')
def build_Lebedev_P2V_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            _Identity()),
            Resize((224,224)),
        ), Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_train_dataloader(LebedevDataset, configs, C)


@DATA.register_func('Lebedev_P2V_eval_dataset')
def build_Lebedev_P2V_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Resize((224,224)),
        ), Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_eval_dataloader(LebedevDataset, configs)


@DATA.register_func('Lebedev_CiDL_train_dataset')
def build_Lebedev_CiDL_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(127.5, 127.5), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_train_dataloader(LebedevDataset, configs, C)


@DATA.register_func('Lebedev_CiDL_eval_dataset')
def build_Lebedev_CiDL_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(127.5, 127.5), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_eval_dataloader(LebedevDataset, configs)


@DATA.register_func('Lebedev_DnD_train_dataset')
def build_Lebedev_DnD_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            _Identity()),
            Resize((224,224)),
        ), Normalize(mu=np.array([123.675, 116.28, 103.53]), sigma=np.array([58.395, 57.12, 57.375])), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_train_dataloader(LebedevDataset, configs, C)


@DATA.register_func('Lebedev_DnD_eval_dataset')
def build_Lebedev_DnD_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(Compose(
            Resize((224,224)),
        ), Normalize(mu=np.array([123.675, 116.28, 103.53]), sigma=np.array([58.395, 57.12, 57.375])), None),
        root=constants.IMDB_LEBEDEV,
        sets=('real',)
    ))

    from data.lebedev import LebedevDataset
    return build_eval_dataloader(LebedevDataset, configs)