import os.path
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


# Data builder utilities
def build_train_dataloader(cls, configs, C):
    return data.DataLoader(
        cls(**configs),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=C['num_workers'],
        pin_memory=C['device']!='cpu',
        drop_last=True
    )


def build_eval_dataloader(cls, configs):
    return data.DataLoader(
        cls(**configs),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )


def get_common_train_configs(C):
    return dict(phase='train', repeats=C['repeats'])


def get_common_eval_configs(C):
    return dict(phase='eval', transforms=[None, None, None], subset=C['subset'])


# Dataset prototype
class DatasetBase(data.Dataset, metaclass=ABCMeta):
    def __init__(
        self, 
        root, phase,
        transforms,
        repeats, 
        subset
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            raise FileNotFoundError
        # phase stands for the working mode,
        # 'train' for training and 'eval' for validating or testing.
        # if phase not in ('train', 'eval'):
        #     raise ValueError("Invalid phase")
        # subset is the sub-dataset to use.
        # For some datasets there are three subsets,
        # while for others there are only train and test(val).
        # if subset not in ('train', 'val', 'test'):
        #     raise ValueError("Invalid subset")
        self.phase = phase
        self.transforms = transforms
        self.repeats = repeats
        # Use 'train' subset during training.
        self.subset = 'train' if self.phase == 'train' else subset

    def __len__(self):
        return self.len * self.repeats

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        index = index % self.len

        item = self.fetch_and_preprocess(index)

        return item

    @abstractmethod
    def fetch_and_preprocess(self, index):
        return None
