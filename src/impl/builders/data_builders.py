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


@DATA.register_func('AC-Szada_eval_dataset')
def build_ac_szada_eval_dataset(C):
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


@DATA.register_func('AC-Tiszadob_eval_dataset')
def build_ac_tiszadob_eval_dataset(C):
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
def build_oscd_train_dataset(C):
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
def build_oscd_eval_dataset(C):
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
        

@DATA.register_func('SVCD_train_dataset')
def build_svcd_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Compose(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            Identity()),
        ), Normalize(0.0, 255.0), None),
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
        Normalize(0.0, 255.0), None),
        root=constants.IMDB_SVCD,
        sets=('real',)
    ))

    from data.svcd import SVCDDataset
    return build_eval_dataloader(SVCDDataset, configs)


@DATA.register_func('SVCD_P2V_train_dataset')
def build_svcd_p2v_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            Identity()), 
            Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_SVCD,
        sets=('real',)
    ))

    from data.svcd import SVCDDataset
    return build_train_dataloader(SVCDDataset, configs, C)


@DATA.register_func('SVCD_P2V_eval_dataset')
def build_svcd_p2v_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_SVCD,
        sets=('real',)
    ))

    from data.svcd import SVCDDataset
    return build_eval_dataloader(SVCDDataset, configs)


@DATA.register_func('LEVIRCD_P2V_train_dataset')
def build_levircd_p2v_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            Identity()), Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_LEVIRCD,
    ))

    from data.levircd import LEVIRCDDataset
    return build_train_dataloader(LEVIRCDDataset, configs, C)


@DATA.register_func('LEVIRCD_P2V_eval_dataset')
def build_levircd_p2v_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_LEVIRCD,
    ))

    from data.levircd import LEVIRCDDataset
    return build_eval_dataloader(LEVIRCDDataset, configs)


@DATA.register_func('WHU_P2V_train_dataset')
def build_whu_p2v_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Choose(
            HorizontalFlip(), VerticalFlip(), 
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(), 
            Identity()), Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_WHU,
    ))

    from data.whu import WHUDataset
    return build_train_dataloader(WHUDataset, configs, C)


@DATA.register_func('WHU_P2V_eval_dataset')
def build_whu_p2v_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_WHU,
    ))

    from data.whu import WHUDataset
    return build_eval_dataloader(WHUDataset, configs)


@DATA.register_func('SYSUCD_P2V_train_dataset')
def build_sysucd_p2v_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(
            Choose(
                HorizontalFlip(), VerticalFlip(), 
                Rotate('90'), Rotate('180'), Rotate('270'),
                Shift(), 
                Identity()), 
            Compose(
                # ContrastBrightScale(),
                # AddGaussNoise(sigma=5.0, prob_apply=0.2),
                Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195]))),
            None),
        root=constants.IMDB_SYSUCD,
    ))

    from data.sysucd import SYSUCDDataset
    return build_train_dataloader(SYSUCDDataset, configs, C)


@DATA.register_func('SYSUCD_P2V_eval_dataset')
def build_sysucd_p2v_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(mu=np.array([110.2008, 100.63983, 95.99475]), sigma=np.array([58.14765, 56.46975, 55.332195])), None),
        root=constants.IMDB_SYSUCD,
    ))

    from data.sysucd import SYSUCDDataset
    return build_eval_dataloader(SYSUCDDataset, configs)