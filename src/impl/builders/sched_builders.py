# Scheduler builders

from collections.abc import Iterable

import torch
import torch.optim.lr_scheduler

from core.misc import Registry


SCHEDS = Registry()


def build_schedulers(cfg_list, optimizers):
    if not isinstance(optimizers, Iterable):
        optimizers = [optimizers]
    schedulers = []
    if len(cfg_list) != len(optimizers):
        raise ValueError("The number of schedulers does not match the number of optimizers.")
    for cfg, optim in zip(cfg_list, optimizers):
        name = cfg.pop('name')
        try:
            sched_cls = getattr(torch.optim.lr_scheduler, name)
            sched_obj = sched_cls(optim, **cfg)
        except AttributeError:
            try:
                builder_name = '_'.join([name, 'sched'])
                sched_obj = SCHEDS[builder_name](optim, cfg)
            except KeyError:
                raise NotImplementedError("{} is not a supported scheduler type.".format(name))
        schedulers.append(sched_obj)
    return schedulers