import os
import os.path as osp
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from tqdm import tqdm

from core.trainer import Trainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    normalize_minmax, normalize_8bit,
    quantize_8bit as quantize,
)
from utils.utils import build_schedulers, HookHelper, FeatureContainer
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score)


class CDTrainer(Trainer):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)

        # Set up tensorboard
        self.tb_on = (hasattr(self.logger, 'log_path') or self.debug) and self.ctx['tb_on']
        if self.tb_on:
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
            self.logger.show_nl("TensorBoard logdir: {}\n".format(osp.abspath(self.gpc.get_dir('tb'))))
            self.tb_intvl = self.ctx['tb_intvl']
            
            # Global steps
            self.train_step = 0
            self.eval_step = 0

        # Whether to save network output
        self.save = self.ctx['save_on'] and not self.debug
        if self.save: 
            self._mt_pool = ThreadPoolExecutor(max_workers=2)
        self.out_dir = self.ctx['out_dir']

        # Build lr schedulers
        self.sched_on = self.ctx['sched_on'] and self.is_training
        if self.sched_on:
            self.schedulers = build_schedulers(self.ctx['schedulers'], self.optimizer)

        self._init_trainer()
            
    def init_learning_rate(self):
        if not self.sched_on:
            self.lr = super().init_learning_rate()
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
            self.lr = self.optimizer.param_groups[0]['lr']
        return self.lr

    def adjust_learning_rate(self, epoch, acc):
        if not self.sched_on:
            self.lr = super().adjust_learning_rate(epoch, acc)
        else:
            for sched in self.schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(acc)
                else:
                    sched.step()
            self.lr = self.optimizer.param_groups[0]['lr']
        return self.lr

    def train_epoch(self, epoch):
        losses = Meter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        
        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = self._prepare_data(t1, t2, tar)

            show_imgs_on_tb = self.tb_on and (i%self.tb_intvl == 0)
            
            fetch_dict = self._set_fetch_dict()
            out_dict = FeatureContainer()

            with HookHelper(self.model, fetch_dict, out_dict, hook_type='forward_out'):
                out = self.model(t1, t2)

            pred = self._process_model_out(out)
            
            loss = self.criterion(pred, tar)
            losses.update(loss.item(), n=tar.shape[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_train, losses.val, losses.avg)

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/running_loss", losses.val, self.train_step)
                if show_imgs_on_tb:
                    t1, t2 = to_array(t1[0]), to_array(t2[0])
                    t1, t2 = self._denorm_image(t1), self._denorm_image(t2)
                    t1, t2 = self._process_input_pairs(t1, t2)
                    self.tb_writer.add_image("Train/t1_picked", t1, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/t2_picked", t2, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/labels_picked", to_array(tar[0]), self.train_step, dataformats='HW')
                    for key, feats in out_dict.items():
                        for idx, feat in enumerate(feats):
                            feat = self._process_fetched_feat(feat)
                            self.tb_writer.add_image(f"Train/{key}_{idx}", feat, self.train_step, dataformats='HWC')
                    self.tb_writer.flush()
                self.train_step += 1
        
        if self.tb_on:
            self.tb_writer.add_scalar("Train/loss", losses.avg, self.train_step)
            self.tb_writer.add_scalar("Train/lr", self.lr, self.train_step)

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = Meter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), Accuracy(mode='accum'))

        self.model.eval()

        with torch.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                t1, t2, tar = self._prepare_data(t1, t2, tar)
                batch_size = tar.shape[0]

                fetch_dict = self._set_fetch_dict()
                out_dict = FeatureContainer()

                with HookHelper(self.model, fetch_dict, out_dict, hook_type='forward_out'):
                    out = self.model(t1, t2)

                pred = self._process_model_out(out)

                loss = self.criterion(pred, tar)
                losses.update(loss.item(), n=batch_size)

                # Convert to numpy arrays
                prob = self._pred_to_prob(pred)
                prob = prob.cpu().numpy()
                cm = (prob>0.5).astype('uint8')
                tar = tar.cpu().numpy().astype('uint8')

                for m in metrics:
                    m.update(cm, tar, n=batch_size)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_eval, losses.val, losses.avg)
                for m in metrics:
                    desc += " {} {:.4f}".format(m.__name__, m.val)

                pb.set_description(desc)
                dump = not self.is_training or (i % max(1, len_eval//10) == 0)
                if dump:
                    self.logger.dump(desc)

                if self.tb_on:
                    if dump:
                        for j in range(batch_size):
                            t1_, t2_ = to_array(t1[j]), to_array(t2[j])
                            t1_, t2_ = self._denorm_image(t1_), self._denorm_image(t2_)
                            t1_, t2_ = self._process_input_pairs(t1_, t2_)
                            self.tb_writer.add_image("Eval/t1", t1_, self.eval_step, dataformats='HWC')
                            self.tb_writer.add_image("Eval/t2", t2_, self.eval_step, dataformats='HWC')
                            self.tb_writer.add_image("Eval/labels", quantize(tar[j]), self.eval_step, dataformats='HW')
                            self.tb_writer.add_image("Eval/prob", to_pseudo_color(quantize(prob[j])), self.eval_step, dataformats='HWC')
                            self.tb_writer.add_image("Eval/cm", quantize(cm[j]), self.eval_step, dataformats='HW')
                            for key, feats in out_dict.items():
                                for idx, feat in enumerate(feats):
                                    feat = self._process_fetched_feat(feat[j])
                                    self.tb_writer.add_image(f"Eval/{key}_{idx}", feat, self.eval_step, dataformats='HWC')
                            self.eval_step += 1
                    else:
                        self.eval_step += batch_size
                
                if self.save:
                    for j in range(batch_size):
                        self.save_image(name[j], quantize(cm[j]), epoch)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            for m in metrics:
                self.tb_writer.add_scalar(f"Eval/{m.__name__.lower()}", m.val, self.eval_step)
            self.tb_writer.flush()

        return metrics[2].val   # F1-score

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
        return self._mt_pool.submit(partial(io.imsave, check_contrast=False), out_path, image)

    def _denorm_image(self, x):
        return x*np.asarray(self.ctx['sigma']) + np.asarray(self.ctx['mu'])

    def _process_input_pairs(self, t1, t2):
        vis_band_inds = self.ctx['tb_vis_bands']
        t1 = t1[...,vis_band_inds]
        t2 = t2[...,vis_band_inds]
        if self.ctx['tb_vis_norm'] == '8bit':
            t1 = normalize_8bit(t1)
            t2 = normalize_8bit(t2)
        else:
            t1 = normalize_minmax(t1)
            t2 = normalize_minmax(t2)
        t1 = np.clip(t1, 0.0, 1.0)
        t2 = np.clip(t2, 0.0, 1.0)
        return t1, t2

    def _process_fetched_feat(self, feat):
        feat = normalize_minmax(feat.mean(0))
        feat = quantize(to_array(feat))
        feat = to_pseudo_color(feat)
        return feat

    def _init_trainer(self):
        pass

    def _prepare_data(self, t1, t2, tar):
        return t1.to(self.device), t2.to(self.device), tar.to(self.device)

    def _set_fetch_dict(self):
        return dict()

    def _process_model_out(self, out):
        return out

    def _pred_to_prob(self, pred):
        return torch.nn.functional.sigmoid(pred)