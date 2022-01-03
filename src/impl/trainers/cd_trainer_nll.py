import torch

from .cd_trainer import CDTrainer


class CDTrainer_NLL(CDTrainer):
    def _process_model_out(self, out):
        return torch.nn.functional.log_softmax(out, dim=1)

    def _pred_to_prob(self, pred):
        return torch.exp(pred[:,1])