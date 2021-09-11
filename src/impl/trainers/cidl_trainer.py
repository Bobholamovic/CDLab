import torch
from tqdm import tqdm

from .cd_trainer import CDTrainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    quantize_8bit as quantize
)
from utils.utils import HookHelper
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score)


class CiDLTrainer(CDTrainer):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)
        self.lambda_adv = settings['lambda_adv']
        self.lambda_vis = settings['lambda_vis']
        self.lambda_lat = settings['lambda_lat']
        self.lambda_self = settings['lambda_self']
        self.lambda_cross = settings['lambda_cross']
        self.thresh = settings['threshold']

    def train_epoch(self, epoch):
        losses = Meter()
        losses_d = Meter()
        losses_adv, losses_vis, losses_lat, losses_self, losses_cross = Meter(), Meter(), Meter(), Meter(), Meter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        
        self.model.train()

        enc_con, enc_sty, dec, dis1, dis2 = self.model
        critn_adv, critn_vis, critn_lat, critn_self, critn_cross = self.criterion
        enc_con_optim, enc_sty_optim, dec_optim, dis1_optim, dis2_optim = self.optimizer
        
        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
            
            con1 = enc_con.forward_a(t1)
            sty1 = enc_sty.forward_a(t1)
            con2 = enc_con.forward_b(t2)
            sty2 = enc_sty.forward_b(t2)
            
            y_c1s2 = dec.forward_a(con1, sty2)
            y_c2s1 = dec.forward_b(con2, sty1)
            
            dout_21 = dis1(y_c2s1.detach())[0]
            ones = torch.ones_like(dout_21)
            zeros = torch.zeros_like(dout_21)
            loss_d = 0.5 * (critn_adv(dout_21, zeros) + critn_adv(dis1(t1)[0], ones)) + \
                0.5 * (critn_adv(dis2(y_c1s2.detach())[0], zeros) + critn_adv(dis2(t2)[0], ones))

            losses_d.update(loss_d.item(), n=self.batch_size)

            # Update D
            dis1_optim.zero_grad()
            dis2_optim.zero_grad()
            loss_d.backward()
            dis1_optim.step()
            dis2_optim.step()

            loss_adv = critn_adv(dis1(y_c2s1)[0], ones) + critn_adv(dis2(y_c1s2)[0], ones)

            y_c1s1 = dec.forward_a(con1, sty1)
            y_c2s2 = dec.forward_b(con2, sty2)

            con12 = enc_con.forward_b(y_c1s2)
            sty12 = enc_sty.forward_b(y_c1s2)
            con21 = enc_con.forward_a(y_c2s1)
            sty21 = enc_sty.forward_a(y_c2s1)

            y_121 = dec.forward_b(con12, sty21)
            y_212 = dec.forward_a(con21, sty12)
            
            loss_vis = critn_vis(y_121, t1) + critn_vis(y_212, t2)
            loss_lat = critn_lat(con12, con1) + critn_lat(con21, con2) + \
                        critn_lat(sty21, sty1) + critn_lat(sty12, sty2)
            loss_self = critn_self(y_c1s1, t1) + critn_self(y_c2s2, t2)
            loss_cross = critn_cross(tar, y_c2s1, t1) + critn_cross(tar, y_c1s2, t2)
            
            loss = self.lambda_adv * loss_adv + self.lambda_vis * loss_vis + self.lambda_lat * loss_lat + \
                self.lambda_self * loss_self + self.lambda_cross * loss_cross

            losses.update(loss.item(), n=self.batch_size)
            losses_adv.update(loss_adv.item(), n=self.batch_size)
            losses_vis.update(loss_vis.item(), n=self.batch_size)
            losses_lat.update(loss_lat.item(), n=self.batch_size)
            losses_self.update(loss_self.item(), n=self.batch_size)
            losses_cross.update(loss_cross.item(), n=self.batch_size)

            # Update G
            enc_con_optim.zero_grad()
            enc_sty_optim.zero_grad()
            dec_optim.zero_grad()
            loss.backward()
            enc_con_optim.zero_grad()
            enc_sty_optim.zero_grad()
            dec_optim.zero_grad()

            desc = (start_pattern+" Loss_G: {:.4f} ({:.4f}) Loss_D: {:.4f} ({:.4f})").format(
                i+1, len_train, 
                losses.val, losses.avg,
                losses_d.val, losses_d.avg
            )

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/running_loss_d", losses_d.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss", losses.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_adv", losses_adv.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_vis", losses_vis.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_lat", losses_lat.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_self", losses_self.val, self.train_step)
                self.tb_writer.add_scalar("Train/running_loss_cross", losses_cross.val, self.train_step)
                self.train_step += 1

        if self.tb_on:
            self.tb_writer.add_scalar("Train/loss_d", losses_d.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss", losses.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_adv", losses_adv.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_vis", losses_vis.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_lat", losses_lat.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_self", losses_self.avg, self.train_step)
            self.tb_writer.add_scalar("Train/loss_cross", losses_cross.avg, self.train_step)

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), Accuracy(mode='accum'))

        self.model.eval()

        enc_con, enc_sty, dec = self.model[:3]

        with torch.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
                
                con1 = enc_con.forward_a(t1)
                sty1 = enc_sty.forward_a(t1)
                con2 = enc_con.forward_b(t2)
                sty2 = enc_sty.forward_b(t2)
                
                y_c1s2 = dec.forward_a(con1, sty2)
                y_c2s1 = dec.forward_b(con2, sty1)

                pred = torch.sqrt(torch.pow(y_c2s1-t1, 2).sum(1)) + torch.sqrt(torch.pow(y_c1s2-t2, 2).sum(1))
                pred -= pred.min()
                pred /= pred.max()

                # Convert to numpy arrays
                cm = to_array(pred[0]>self.thresh).astype('uint8')
                tar = to_array(tar[0]).astype('uint8')

                for m in metrics:
                    m.update(cm, tar)

                desc = start_pattern.format(i+1, len_eval)
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
                        y_c1s2 = self.denorm(to_array(y_c1s2[0])).astype('uint8')
                        y_c2s1 = self.denorm(to_array(y_c2s1[0])).astype('uint8')
                        self.tb_writer.add_image("Eval/y_c1s2", y_c1s2, self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/y_c2s1", y_c2s1, self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/labels", quantize(tar), self.eval_step, dataformats='HW')
                        prob = quantize(to_array(pred[0]))
                        self.tb_writer.add_image("Eval/prob", to_pseudo_color(prob), self.eval_step, dataformats='HWC')
                        self.tb_writer.add_image("Eval/cm", quantize(cm), self.eval_step, dataformats='HW')
                    self.eval_step += 1
                
                if self.save:
                    self.save_image(name[0], quantize(cm), epoch)

        if self.tb_on:
            self.tb_writer.add_scalars("Eval/metrics", {m.__name__.lower(): m.val for m in metrics}, self.eval_step)
            self.tb_writer.flush()

        return metrics[2].val   # F1-score