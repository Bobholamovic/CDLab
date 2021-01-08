from core.misc import R
from .cd_trainer import CDTrainer
from .cd_trainer_nll import CDTrainer_NLL

__all__ = []

trainer_switcher = R['Trainer_switcher']
trainer_switcher.add_item(lambda C: C['criterion']=='WNLL' and (not C['tb_on'] or C['dataset'] != 'OSCD'), CDTrainer_NLL)