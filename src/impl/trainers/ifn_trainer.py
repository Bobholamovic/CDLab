import torch
import torch.nn.functional as F

from .cd_trainer_bce import CDTrainer_BCE
from utils.losses import MixedLoss, CombinedLoss


class IFNTrainer(CDTrainer_BCE):
    def _init_trainer(self):
        if self.ctx.get('mix_coeffs') is not None:
            self.criterion = MixedLoss(self.criterion, self.ctx['mix_coeffs'])
        if self.ctx.get('cmb_coeffs') is not None:
            self.criterion = CombinedLoss(self.criterion, self.ctx['cmb_coeffs'])

    def _process_model_out(self, out):
        size = out[0].shape[2:]
        return [F.interpolate(o, size=size).squeeze(1) for o in out]

    def _pred_to_prob(self, pred):
        return F.sigmoid(pred[0])