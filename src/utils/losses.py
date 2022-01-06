import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedLoss(nn.Module):
    def __init__(self, critns, coeffs=1.0):
        super().__init__()
        self.critns = critns
        if isinstance(coeffs, float):
            coeffs = [coeffs]*len(critns)
        if len(coeffs) != len(critns):
            raise ValueError
        self.coeffs = coeffs

    def forward(self, pred, tar):
        loss = 0.0
        for critn, coeff in zip(self.critns, self.coeffs):
            loss += coeff * critn(pred, tar)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, critn, coeffs=1.0):
        super().__init__()
        self.critn = critn
        self.coeffs = coeffs

    def forward(self, preds, tar):
        if isinstance(self.coeffs, float):
            coeffs = [self.coeffs]*len(preds)
        else:
            coeffs = self.coeffs
        if len(coeffs) != len(preds):
            raise ValueError
        loss = 0.0
        for coeff, pred in zip(coeffs, preds):
            loss += coeff * self.critn(pred, tar)
        return loss


class CombinedLoss_DS(nn.Module):
    def __init__(self, critn_main, critn_aux, coeff_main=1.0, coeffs_aux=1.0, main_idx=0):
        super().__init__()
        self.critn_main = critn_main
        self.critn_aux = critn_aux
        self.coeff_main = coeff_main
        self.coeffs_aux = coeffs_aux
        self.main_idx = main_idx

    def forward(self, preds, tar, tar_aux=None):
        if tar_aux is None:
            tar_aux = tar

        pred_main = preds[self.main_idx]
        preds_aux = [pred for i, pred in enumerate(preds) if i != self.main_idx]

        if isinstance(self.coeffs_aux, float):
            coeffs_aux = [self.coeffs_aux]*len(preds_aux)
        else:
            coeffs_aux = self.coeffs_aux
        if len(coeffs_aux) != len(preds_aux):
            raise ValueError

        loss = self.critn_main(pred_main, tar)
        for coeff, pred in zip(coeffs_aux, preds_aux):
            loss += coeff * self.critn_aux(pred, tar_aux)
        return loss


# Refer to https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, pred, tar):
        pred, tar = pred.flatten(1), tar.flatten(1)
        prob = F.sigmoid(pred)

        num = (prob*tar).sum(1) + self.smooth
        den = (prob.pow(self.p) + tar.pow(self.p)).sum(1) + self.smooth

        loss = 1 - num/den
        
        return loss.mean()


class BCLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.m = margin
        self.eps = 1e-4

    def forward(self, pred, tar):
        utar = 1-tar
        n_u = utar.sum() + self.eps
        n_c = tar.sum() + self.eps
        loss = 0.5*torch.sum(utar*torch.pow(pred, 2)) / n_u + \
            0.5*torch.sum(tar*torch.pow(torch.clamp(self.m-pred, min=0.0), 2)) / n_c
        return loss
