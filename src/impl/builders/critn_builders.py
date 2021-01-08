# Custom criterion builders

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.misc import CRITNS


@CRITNS.register_func('WNLL_critn')
def build_WeightedNLL_critn(C):
    return nn.NLLLoss(torch.Tensor(C['weights']))