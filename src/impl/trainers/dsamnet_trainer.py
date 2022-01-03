import torch
import torch.nn.functional as F

from core.factories import critn_factory
from .cd_trainer import CDTrainer
from utils.losses import CombinedLoss_DS


class DSAMNetTrainer(CDTrainer):
    def _init_trainer(self):
        self.thresh = self.ctx['threshold']
        lambda_ = self.ctx['lambda']
        aux_critn = critn_factory(self.ctx['aux_critn'], self.ctx)
        self.criterion = CombinedLoss_DS(self.criterion, aux_critn, coeff_main=1.0, coeffs_aux=0.5*lambda_)

    def _prepare_data(self, t1, t2, tar):
        return super()._prepare_data(t1, t2, tar.float())

    def _process_model_out(self, out):
        return [o.squeeze(1) for o in out]

    def _pred_to_prob(self, pred):
        return (pred[0]>self.thresh).float()