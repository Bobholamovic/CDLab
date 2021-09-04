import torch
from tqdm import tqdm

from .cd_trainer import CDTrainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    quantize_8bit as quantize
)
from utils.utils import HookHelper
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score)
from utils.data_utils.augmentations import Compose
from utils.data_utils.preprocessors import Normalize


class I2VTrainer(CDTrainer):
    def __init__(self, settings):
        assert settings['model'] == 'I2V'
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)
        self.lambda_i = settings['lambda_i']
        self.lambda_v = settings['lambda_v']
        self.thresh = settings['threshold']

    def train_epoch(self, epoch):
        losses = Meter()
        losses_i, losses_v = Meter(), Meter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)

        critn_i, critn_v = self.criterion
        
        self.model.train()
        
        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
            tar = tar.float()
            
            show_imgs_on_tb = self.tb_on and (i%self.tb_intvl == 0)
            
            pred_i, pred_v = self.model(t1, t2)
            pred_i = pred_i.squeeze(1)
            pred_v = pred_v.squeeze(1)
            
            loss_i = critn_i(pred_i, tar)
            loss_v = critn_v(pred_v, tar)
            loss = loss_i * self.lambda_i + loss_v * self.lambda_v
            losses_i.update(loss_i.item(), n=self.batch_size)
            losses_v.update(loss_v.item(), n=self.batch_size)
            losses.update(loss.item(), n=self.batch_size)
            
            losses.update(loss.item(), n=self.batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            desc = (start_pattern+" Loss: {:.4f} ({:.4f}) Loss_i: {:.4f} ({:.4f}) Loss_v: {:.4f} ({:.4f})").format(
                    i+1, len_train, 
                    losses.val, losses.avg,
                    losses_i.val, losses_i.avg,
                    losses_v.val, losses_v.avg
                )

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/running_loss", losses.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_i", losses_i.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_v", losses_v.val, self.train_step)
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
            self.tb_writer.add_scalar("Train/loss_i", losses_i.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_v", losses_v.avg, self.train_step)

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = Meter()
        losses_i, losses_v = Meter(), Meter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(), Recall(), F1Score(), Accuracy())

        critn_i, critn_v = self.criterion

        self.model.eval()

        with torch.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
                tar = tar.float()
                
                pred_i, pred_v = self.model(t1, t2)
                pred_i = pred_i.squeeze(1)
                pred_v = pred_v.squeeze(1)

                loss_i = critn_i(pred_i, tar)
                loss_v = critn_v(pred_v, tar)
                loss = loss_i * self.lambda_i + loss_v * self.lambda_v
                losses_i.update(loss_i.item(), n=self.batch_size)
                losses_v.update(loss_v.item(), n=self.batch_size)
                losses.update(loss.item(), n=self.batch_size)

                # Convert to numpy arrays
                cm_i = to_array(torch.sigmoid(pred_i[0])>self.thresh).astype('uint8')
                cm_v = to_array(torch.sigmoid(pred_v[0])>self.thresh).astype('uint8')
                tar = to_array(tar[0]).astype('uint8')

                for m in metrics:
                    m.update(cm_v, tar)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f}) Loss_i: {:.4f} ({:.4f}) Loss_v: {:.4f} ({:.4f})").format(
                    i+1, len_eval, 
                    losses.val, losses.avg,
                    losses_i.val, losses_i.avg,
                    losses_v.val, losses_v.avg
                )
                for m in metrics:
                    desc += " {} {:.4f} ({:.4f})".format(m.__name__, m.val, m.avg)

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
                        prob_i = quantize(to_array(torch.sigmoid(pred_i)))
                        self.tb_writer.add_image("Eval/prob_i", to_pseudo_color(prob_i), self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/cm_i", quantize(cm_i), self.eval_step, dataformats='HW')
                        prob_v = quantize(to_array(torch.sigmoid(pred_v)))
                        self.tb_writer.add_image("Eval/prob_v", to_pseudo_color(prob_v), self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/cm_v", quantize(cm_v), self.eval_step, dataformats='HW')
                    self.eval_step += 1
                
                if self.save:
                    self.save_image(name[0], quantize(cm_v), epoch)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            self.tb_writer.add_scalar("Eval/loss_i", losses_i.avg, self.eval_step)
            self.tb_writer.add_scalar("Eval/loss_v", losses_v.avg, self.eval_step)
            self.tb_writer.add_scalars("Eval/metrics", {m.__name__.lower(): m.avg for m in metrics}, self.eval_step)
            self.tb_writer.flush()

        return metrics[2].avg   # F1-score
    
    def denorm(self, x):
        # HACK: perhaps I should consider norm and denorm in the design
        def _make_denorm_func(norm_tf):
            assert not norm_tf.zscore
            return lambda x: x * norm_tf.sigma + norm_tf.mu

        transforms = self.train_loader.dataset.transforms[1] if self.is_training else self.eval_loader.dataset.transforms[1]
        if isinstance(transforms, Compose):
            norm_tfs = filter(lambda tf: isinstance(tf, Normalize), transforms.tfs)
            try:
                norm_tf = next(norm_tfs)
            except StopIteration:
                raise ValueError
            denorm_func = _make_denorm_func(norm_tf)
            if next(norm_tf, None) is not None:
                raise ValueError
        elif isinstance(transforms, Normalize):
            denorm_func = _make_denorm_func(transforms)
        else:
            raise ValueError
        return denorm_func(x)