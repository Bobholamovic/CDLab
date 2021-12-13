from core.misc import R
from .cd_trainer import CDTrainer
from .cd_trainer_nll import CDTrainer_NLL
from .p2v_trainer import P2VTrainer


__all__ = []

trainer_switcher = R['Trainer_switcher']
trainer_switcher.add_item(lambda C: C['criterion']=='WNLL' and (not C['tb_on'] or C['dataset'] != 'OSCD'), CDTrainer_NLL)
trainer_switcher.add_item(lambda C: C['model'].startswith('P2V'), P2VTrainer)