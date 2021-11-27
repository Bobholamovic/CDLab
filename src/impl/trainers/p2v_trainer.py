import os.path as osp
import operator

import torch
from tqdm import tqdm

from .cd_trainer import CDTrainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    quantize_8bit as quantize,
    save_gif
)
from utils.utils import HookHelper, FeatureContainer
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score)


class P2VTrainer(CDTrainer):
    def __init__(self, settings):
        assert settings['model'] == 'P2V'
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)
        self.thresh = settings['threshold']
        self.lambda_ = settings['lambda']

    def train_epoch(self, epoch):
        losses = Meter()
        losses_aux_v = Meter()
        losses_aux_p = Meter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        
        for i, (t1, t2, tar) in enumerate(pb):
            # if tar.sum() == 0:
            #     continue
            t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
            tar = tar.float()
            
            show_imgs_on_tb = self.tb_on and (i%self.tb_intvl == 0)
            
            pred, pred_v, pred_p = self.model(t1, t2, return_aux=True)

            loss = self.criterion(pred.squeeze(1), tar)
            loss_aux_v = self.criterion(pred_v.squeeze(1), tar)
            loss_aux_p = self.criterion(pred_p.squeeze(1), tar)
            losses.update(loss.item(), n=self.batch_size)
            losses_aux_v.update(loss_aux_v.item(), n=self.batch_size)
            losses_aux_p.update(loss_aux_p.item(), n=self.batch_size)

            self.optimizer.zero_grad()
            (loss+self.lambda_*loss_aux_v+self.lambda_*loss_aux_p).backward()
            self.optimizer.step()

            desc = (start_pattern+" Loss: {:.4f} ({:.4f}) Loss_aux_v: {:.4f} ({:.4f}) Loss_aux_p: {:.4f} ({:.4f})").format(
                    i+1, len_train, 
                    losses.val, losses.avg,
                    losses_aux_v.val, losses_aux_v.avg,
                    losses_aux_p.val, losses_aux_p.avg,
                )

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/running_loss", losses.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_aux_v", losses_aux_v.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_aux_p", losses_aux_p.val, self.train_step)
                if show_imgs_on_tb:
                    t1 = self.denorm(to_array(t1.detach()[0])).astype('uint8')
                    t2 = self.denorm(to_array(t2.detach()[0])).astype('uint8')
                    self.tb_writer.add_image("Train/t1_picked", t1, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/t2_picked", t2, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/labels_picked", tar[0].unsqueeze(0), self.train_step)
                    self.tb_writer.flush()
                self.train_step += 1
            
        if self.tb_on:
            self.tb_writer.add_scalar("Train/loss", losses.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_aux_v", losses_aux_v.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_aux_p", losses_aux_p.avg, self.train_step)

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = Meter()
        losses_aux_v = Meter()
        losses_aux_p = Meter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), Accuracy(mode='accum'))

        self.model.eval()

        with torch.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
                tar = tar.float()
                
                if self.tb_on or self.save:
                    feats = FeatureContainer()
                    fetch_dict_fi = {
                        'encoder_v': 'frames'
                    }
                    fetch_dict_fo = {}
                    with HookHelper(self.model, fetch_dict_fi, feats, 'forward_in'), \
                        HookHelper(self.model, fetch_dict_fo, feats, 'forward_out'):
                        pred, pred_v, pred_p = self.model(t1, t2, return_aux=True)
                else:
                    pred, pred_v, pred_p = self.model(t1, t2, return_aux=True)
                
                pred = pred.squeeze(1)

                loss = self.criterion(pred, tar)
                loss_aux_v = self.criterion(pred_v.squeeze(1), tar)
                loss_aux_p = self.criterion(pred_p.squeeze(1), tar)
                losses.update(loss.item())
                losses_aux_v.update(loss_aux_v.item())
                losses_aux_p.update(loss_aux_p.item())

                # Convert to numpy arrays
                cm = to_array(torch.sigmoid(pred[0])>self.thresh).astype('uint8')
                tar = to_array(tar[0]).astype('uint8')

                for m in metrics:
                    m.update(cm, tar)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f}) Loss_aux_v: {:.4f} ({:.4f}) Loss_aux_p: {:.4f} ({:.4f})").format(
                    i+1, len_eval, 
                    losses.val, losses.avg,
                    losses_aux_v.val, losses_aux_v.avg,
                    losses_aux_p.val, losses_aux_p.avg
                )
                for m in metrics:
                    desc += " {} {:.4f}".format(m.__name__, m.val)

                pb.set_description(desc)
                dump = not self.is_training or (i % max(1, len_eval//10) == 0)
                if dump:
                    self.logger.dump(desc)
                
                if self.tb_on:
                    if dump:
                        t1 = self.denorm(to_array(t1[0])).astype('uint8')
                        t2 = self.denorm(to_array(t2[0])).astype('uint8')
                        self.tb_writer.add_image("Eval/t1", t1, self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/t2", t2, self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/labels", quantize(tar), self.eval_step, dataformats='HW')
                        prob = quantize(to_array(torch.sigmoid(pred)))
                        self.tb_writer.add_image("Eval/prob", to_pseudo_color(prob), self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/cm", quantize(cm), self.eval_step, dataformats='HW')
                    self.eval_step += 1
                
                if self.save:
                    self.save_image(name[0], quantize(cm), epoch)
                    # if 'frames' in feats.keys():
                    #     frames = feats['frames'][-1][0].transpose(0,1)  # Show last iteration
                    #     frames = list(map(operator.methodcaller('astype', 'uint8'), map(self.denorm, map(to_array, frames))))
                    #     self.save_gif(name[0], frames, epoch)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            self.tb_writer.add_scalar("Eval/loss_aux_v", losses_aux_v.avg, self.eval_step)
            self.tb_writer.add_scalar("Eval/loss_aux_p", losses_aux_p.avg, self.eval_step)
            self.tb_writer.add_scalars("Eval/metrics", {m.__name__.lower(): m.val for m in metrics}, self.eval_step)
            self.tb_writer.flush()

        return metrics[2].val   # F1-score

    def save_gif(self, file_name, images, epoch):
        file_path = osp.join(
            'epoch_{}'.format(epoch),
            self.out_dir,
            osp.splitext(file_name)[0]+'.gif'
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.ctx['suffix_off'],
            auto_make=True,
            underline=True
        )
        return save_gif(out_path, images)

    # def init_learning_rate(self):
    #     lr = super().init_learning_rate()
    #     for group in self.optimizer.param_groups:
    #         if group['name'] == 'encoder_v':
    #             group['lr'] *= 0.25
    #     return lr