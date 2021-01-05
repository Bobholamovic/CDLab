import os
import os.path as osp
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from skimage import io

from core.trainer import Trainer


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
            
    def init_learning_rate(self):
        # Set learning rate adjustment strategy
        if self.ctx['lr_mode'] == 'const':
            return self.lr
        else:
            def _simple_scheduler_step(self, epoch, acc):
                self.scheduler.step()
                return self.scheduler.get_lr()[0]
            def _scheduler_step_with_acc(self, epoch, acc):
                self.scheduler.step(acc)
                # Only return the lr of the first param group
                return self.optimizer.param_groups[0]['lr']
            lr_mode = self.ctx['lr_mode']
            if lr_mode == 'step':
                self.scheduler = lr_scheduler.StepLR( 
                    self.optimizer, self.ctx['step'], gamma=0.5
                )
                self.adjust_learning_rate = partial(_simple_scheduler_step, self)
            elif lr_mode == 'exp':
                self.scheduler = lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.95
                )
                self.adjust_learning_rate = partial(_simple_scheduler_step, self)
            elif lr_mode == 'plateau':
                if self.load_checkpoint:
                    self.logger.warn("The old state of the lr scheduler will not be restored.")
                self.scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.5, threshold=1e-4
                )
                self.adjust_learning_rate = partial(_scheduler_step_with_acc, self)
                return self.optimizer.param_groups[0]['lr']
            else:
                raise NotImplementedError

            if self.start_epoch > 0:
                # Restore previous state
                # FIXME: This will trigger pytorch warning "Detected call of `lr_scheduler.step()` 
                # before `optimizer.step()`" in pytorch 1.1.0 and later.
                # Perhaps I should store the state of scheduler to a checkpoint file and restore it from disk.
                last_epoch = self.start_epoch
                while self.scheduler.last_epoch < last_epoch:
                    self.scheduler.step()
            return self.scheduler.get_lr()[0]

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