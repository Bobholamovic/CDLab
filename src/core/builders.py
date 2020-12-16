# Built-in builders

import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import (MODELS, OPTIMS, CRITNS, DATA)


# Optimizer builders
@OPTIMS.register_func('Adam_optim')
def build_Adam_optim(params, C):
    return torch.optim.Adam(
        params, 
        betas=(0.9, 0.999),
        lr=C['lr'],
        weight_decay=C['weight_decay']
    )


@OPTIMS.register_func('SGD_optim')
def build_SGD_optim(params, C):
    return torch.optim.SGD(
        params, 
        lr=C['lr'],
        momentum=0.9,
        weight_decay=C['weight_decay']
    )


# Criterion builders
@CRITNS.register_func('L1_critn')
def build_L1_critn(C):
    return nn.L1Loss()


@CRITNS.register_func('MSE_critn')
def build_MSE_critn(C):
    return nn.MSELoss()


@CRITNS.register_func('CE_critn')
def build_CE_critn(C):
    return nn.CrossEntropyLoss(torch.Tensor(C['weights']))


@CRITNS.register_func('NLL_critn')
def build_NLL_critn(C):
    return nn.NLLLoss(torch.Tensor(C['weights']))
