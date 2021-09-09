import os
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter
from skimage import io

from core.trainer import Trainer
from ..builders.sched_builders import build_schedulers
from utils.data_utils.augmentations import Compose
from utils.data_utils.preprocessors import Normalize


class CDTrainer(Trainer):
    def __init__(self, model, dataset, criterion, optimizer, settings):
        super().__init__(model, dataset, criterion, optimizer, settings)
        self.tb_on = (hasattr(self.logger, 'log_path') or self.debug) and self.ctx['tb_on']
        if self.tb_on:
            # Initialize tensorboard
            if hasattr(self.logger, 'log_path'):
                tb_dir = self.path(
                    'log', 
                    osp.join('tb', osp.splitext(osp.basename(self.logger.log_path))[0], '.'), 
                    name='tb', 
                    auto_make=True, 
                    suffix=False
                )
            else:
                tb_dir = self.path(
                    'log', 
                    osp.join('tb', 'debug', '.'), 
                    name='tb', 
                    auto_make=True, 
                    suffix=False
                )
                for root, dirs, files in os.walk(self.gpc.get_dir('tb'), False):
                    for f in files:
                        os.remove(osp.join(root, f))
                    for d in dirs:
                        os.rmdir(osp.join(root, d))
            self.tb_writer = SummaryWriter(tb_dir)
            self.logger.show_nl("Tensorboard logdir: {}\n".format(osp.abspath(self.gpc.get_dir('tb'))))
            self.tb_intvl = self.ctx['tb_intvl']
            
            # Global steps
            self.train_step = 0
            self.eval_step = 0

        # Whether to save network output
        self.out_dir = self.ctx['out_dir']
        self.save = self.ctx['save_on'] and not self.debug

        # Build lr schedulers
        self.sched_on = self.ctx['sched_on'] and self.is_training
        if self.sched_on:
            # NOTE: There is no way to set the schedulers configs from commandline arguments!
            self.schedulers = build_schedulers(self.ctx['schedulers'], self.optimizer)
            
    def init_learning_rate(self):
        if not self.sched_on:
            return super().init_learning_rate()
        else:
            for idx, sched in enumerate(self.schedulers):
                if self.start_epoch > 0:
                    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.logger.warn("The old state of lr scheduler {} will not be restored.".format(idx))
                        continue
                    # Restore previous state
                    # FIXME: This will trigger pytorch warning "Detected call of `lr_scheduler.step()` 
                    # before `optimizer.step()`" in pytorch 1.1.0 and later.
                    # Perhaps I should store the state of scheduler to a checkpoint file and restore it from disk.
                    last_epoch = self.start_epoch
                    while sched.last_epoch < last_epoch:
                        sched.step()
            return self.optimizer.param_groups[0]['lr']

    def adjust_learning_rate(self, epoch, acc):
        if not self.sched_on:
            return super().adjust_learning_rate(epoch, acc)
        else:
            for sched in self.schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(acc)
                else:
                    sched.step()
            return self.optimizer.param_groups[0]['lr']

    def save_image(self, file_name, image, epoch):
        file_path = osp.join(
            'epoch_{}'.format(epoch),
            self.out_dir,
            file_name
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.ctx['suffix_off'],
            auto_make=True,
            underline=True
        )
        return io.imsave(out_path, image)

    # def __del__(self):
    #     if self.tb_on:
    #         self.tb_writer.close()

    def denorm(self, x):
        # HACK: perhaps I should consider norm and denorm in the design
        def _make_denorm_func(norm_tf):
            assert not norm_tf.zscore
            return lambda x: x * norm_tf.sigma + norm_tf.mu

        transforms = self.eval_loader.dataset.transforms[1]
        if isinstance(transforms, Compose):
            norm_tfs = filter(lambda tf: isinstance(tf, Normalize), transforms.tfs)
            try:
                norm_tf = next(norm_tfs)
            except StopIteration:
                raise ValueError
            denorm_func = _make_denorm_func(norm_tf)
            if next(norm_tfs, None) is not None:
                raise ValueError
        elif isinstance(transforms, Normalize):
            denorm_func = _make_denorm_func(transforms)
        else:
            raise ValueError
        return denorm_func(x)