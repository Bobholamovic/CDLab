import torch
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from skimage.segmentation._slic import _enforce_label_connectivity_cython as enforce_connectivity

from .cd_trainer import CDTrainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    quantize_8bit as quantize
)
from utils.utils import HookHelper
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score)


class SSNMixin:
    @staticmethod
    def get_abs_idx_map(Q, ops):
        rel_idx_map = torch.argmax(Q.detach(), dim=1, keepdim=True).int()
        return ops['map_idx'](rel_idx_map)

    @staticmethod
    def get_sp_map(Q, x, abs_idx_map, ops):
        spf_x = ops['map_p2sp'](x, Q)
        return ops['smear'](spf_x, abs_idx_map.detach())

    @staticmethod
    def get_R_recon(Q, R, ops):
        # Remap R
        spf_R = ops['map_p2sp'](R.float(), Q)
        return ops['map_sp2p'](spf_R, Q)


class ESCNetTrainer(CDTrainer, SSNMixin):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)
        self.alpha = settings['alpha']
        self.merge = settings['merge_on']
        self.n_spixels = self.model.ssn.n_spixels
        self.critn_cd, self.critn_cmpct = self.criterion

    def train_epoch(self, epoch):
        losses = Meter()
        losses_cd = Meter()
        losses_cmpct = Meter()
        losses_recon = Meter()

        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        
        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
            R = self.make_onehot(tar)
            
            show_imgs_on_tb = self.tb_on and (i%self.tb_intvl == 0)
            
            pred, pred_ds, Q, ops, cvrted_feats = self.model(t1[:,3:], t2[:,3:], merge=self.merge)
            pred = torch.nn.functional.log_softmax(pred, dim=1)
            pred_ds = torch.nn.functional.log_softmax(pred_ds, dim=1)

            loss_cd = self.critn_cd(pred, tar) + 0.5*self.critn_cd(pred_ds, tar)

            loss_cmpct, loss_recon = 0, 0

            for Q_, ops_, cvrted_feats_ in zip(Q, ops, cvrted_feats):
                idx_map = self.get_abs_idx_map(Q_, ops_)
                spixel_map = self.get_sp_map(Q_, cvrted_feats_, idx_map, ops_)
                R_recon = self.get_R_recon(Q_, R, ops_)
                loss_cmpct += self.critn_cmpct(spixel_map, cvrted_feats_)
                loss_recon += self.calc_recon_loss(R_recon, R)
            
            loss = loss_cd + 0.1 * (loss_recon + self.alpha * loss_cmpct)
            
            losses.update(loss.item(), n=self.batch_size)
            losses_cd.update(loss_cd.item(), n=self.batch_size)
            losses_cmpct.update(loss_cmpct.item(), n=self.batch_size)
            losses_recon.update(loss_recon.item(), n=self.batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            desc = (start_pattern+" Loss: {:.4f} ({:.4f}) CD: {:.4f} ({:.4f}) Cmpct: {:.4f} ({:.4f}) Recon: {:.4f} ({:.4f})").format(
                i+1, len_train, 
                losses.val, losses.avg,
                losses_cd.val, losses_cd.avg,
                losses_cmpct.val, losses_cmpct.avg,
                losses_recon.val, losses_recon.avg
            )

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/loss", losses.val, self.train_step)
                self.tb_writer.add_scalar("Train/loss_cd", losses_cd.val, self.train_step)
                self.tb_writer.add_scalar("Train/loss_cmpct", losses_cmpct.val, self.train_step)
                self.tb_writer.add_scalar("Train/loss_recon", losses_recon.val, self.train_step)
                if show_imgs_on_tb:
                    t1 = to_array(t1[0,:3].detach()).astype('uint8')
                    t2 = to_array(t2[0,:3].detach()).astype('uint8')
                    self.tb_writer.add_image("Train/t1_picked", t1, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/t2_picked", t2, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/labels_picked", tar[0].unsqueeze(0), self.train_step)
                    self.tb_writer.flush()
                self.train_step += 1

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))

        losses = Meter()
        losses_cd = Meter()
        losses_cmpct = Meter()
        losses_recon = Meter()

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
                R = self.make_onehot(tar)

                pred, pred_ds, Q, ops, cvrted_feats = self.model(t1[:,3:], t2[:,3:], merge=self.merge)
                pred = torch.nn.functional.log_softmax(pred, dim=1)
                pred_ds = torch.nn.functional.log_softmax(pred_ds, dim=1)

                loss_cd = self.critn_cd(pred, tar) + 0.5*self.critn_cd(pred_ds, tar)

                loss_cmpct, loss_recon = 0, 0

                for Q_, ops_, cvrted_feats_ in zip(Q, ops, cvrted_feats):
                    idx_map = self.get_abs_idx_map(Q_, ops_)
                    spixel_map = self.get_sp_map(Q_, cvrted_feats_, idx_map, ops_)
                    R_recon = self.get_R_recon(Q_, R, ops_)
                    loss_cmpct += self.critn_cmpct(spixel_map, cvrted_feats_)
                    loss_recon += self.calc_recon_loss(R_recon, R)
                
                loss = loss_cd + 0.1 * (loss_recon + self.alpha * loss_cmpct)
                
                losses.update(loss.item(), n=self.batch_size)
                losses_cd.update(loss_cd.item(), n=self.batch_size)
                losses_cmpct.update(loss_cmpct.item(), n=self.batch_size)
                losses_recon.update(loss_recon.item(), n=self.batch_size)

                # Convert to numpy arrays
                cm = to_array(torch.argmax(pred[0], 0)).astype('uint8')
                tar = to_array(tar[0]).astype('uint8')

                for m in metrics:
                    m.update(cm, tar)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f}) CD: {:.4f} ({:.4f})").format(
                    i+1, len_eval, 
                    losses.val, losses.avg,
                    losses_cd.val, losses_cd.avg
                )
                for m in metrics:
                    desc += " {} {:.4f}".format(m.__name__, m.val)

                pb.set_description(desc)
                dump = not self.is_training or (i % max(1, len_eval//10) == 0)
                if dump:
                    self.logger.dump(desc)

                if self.tb_on:
                    if dump:
                        t1 = to_array(t1[0,:3]).astype('uint8')
                        t2 = to_array(t2[0,:3]).astype('uint8')
                        self.tb_writer.add_image("Eval/t1", self.vis_spixels(t1, Q[0], ops[0]), self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/t2", self.vis_spixels(t2, Q[1], ops[1]), self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/labels", quantize(tar), self.eval_step, dataformats='HW')
                        prob = quantize(to_array(torch.exp(pred[0,1])))
                        self.tb_writer.add_image("Eval/prob", to_pseudo_color(prob), self.eval_step, dataformats='HWC')
                        prob_ds = quantize(to_array(torch.exp(pred_ds[0,1])))
                        self.tb_writer.add_image("Eval/prob_ds", to_pseudo_color(prob_ds), self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/cm", quantize(cm), self.eval_step, dataformats='HW')
                    self.eval_step += 1
                
                if self.save:
                    self.save_image(name[0], quantize(cm), epoch)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            self.tb_writer.add_scalar("Eval/loss_cd", losses_cd.avg, self.eval_step)
            self.tb_writer.add_scalar("Eval/loss_cmpct", losses_cmpct.avg, self.eval_step)
            self.tb_writer.add_scalar("Eval/loss_recon", losses_recon.avg, self.eval_step)
            self.tb_writer.add_scalars("Eval/metrics", {m.__name__.lower(): m.val for m in metrics}, self.eval_step)
            self.tb_writer.flush()

        return metrics[2].val   # F1-score

    @staticmethod
    def calc_recon_loss(R_recon, R, norm=False):
        FLT_MIN = 1e-12
        _, label = torch.max(R, dim=1) # argmax
        if norm:
            # Normalization to ensure the sum along channels equals 1
            R_recon = R_recon / (torch.sum(R_recon, dim=1, keepdim=True)+FLT_MIN)
        return torch.nn.functional.nll_loss(torch.log(torch.clamp(R_recon, FLT_MIN, 1.0)), label)

    @staticmethod
    def make_onehot(labels):
        return torch.stack([(labels==0), (labels==1)], dim=1).long()

    def vis_spixels(self, rgb, Q, ops):
        segment_size = (rgb.shape[0] * rgb.shape[1]) / self.n_spixels
        min_size = int(0.06 * segment_size)
        max_size = int(3 * segment_size)
        idx_map = self.get_abs_idx_map(Q, ops)
        idx_map = enforce_connectivity(to_array(idx_map[0]).astype('int64'), min_size, max_size)
        return mark_boundaries(rgb, idx_map[...,0], color=(1,1,1))
