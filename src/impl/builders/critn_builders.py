# Custom criterion builders

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.misc import CRITNS


@CRITNS.register_func('WNLL_critn')
def build_WeightedNLL_critn(C):
    return nn.NLLLoss(torch.Tensor(C['weights']))


@CRITNS.register_func('WBCE_critn')
def build_WeightedBCE_critn(C):
    assert len(C['weights']) == 2
    pos_weight = C['weights'][1]/C['weights'][0]
    return nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))