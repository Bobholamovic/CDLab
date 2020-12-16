import torch
from tqdm import tqdm

from .cd_trainer import CDTrainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color, 
    normalize_8bit,
    quantize_8bit as quantize
)
from utils.utils import HookHelper
from utils.metrics import (AverageMeter, Precision, Recall, Accuracy, F1Score)


class CDTrainer_NLL(CDTrainer):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)

    def train_epoch(self, epoch):
        losses = AverageMeter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        
        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
            
            show_imgs_on_tb = self.tb_on and (i%self.tb_intvl == 0)
            
            prob = self.model(t1, t2)
            
            loss = self.criterion(prob, tar)
            
            losses.update(loss.item(), n=self.batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_train, losses.val, losses.avg)

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/loss", losses.val, self.train_step)
                if show_imgs_on_tb:
                    self.tb_writer.add_image("Train/t1_picked", t1.detach()[0], self.train_step)
                    self.tb_writer.add_image("Train/t2_picked", t2.detach()[0], self.train_step)
                    self.tb_writer.add_image("Train/labels_picked", tar[0].unsqueeze(0), self.train_step)
                    self.tb_writer.flush()
                self.train_step += 1

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = AverageMeter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(), Recall(), F1Score(), Accuracy())

        self.model.eval()

        with torch.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                if self.is_training and i >= self.val_iters:
                    # This saves time
                    pb.close()
                    self.logger.warn("Evaluation ends early.")
                    break
                t1, t2, tar = t1.to(self.device), t2.to(self.device), tar.to(self.device)
                
                prob = self.model(t1, t2)

                loss = self.criterion(prob, tar)
                losses.update(loss.item(), n=self.batch_size)

                # Convert to numpy arrays
                cm = to_array(torch.argmax(prob[0], 0)).astype('uint8')
                tar = to_array(tar[0]).astype('uint8')

                for m in metrics:
                    m.update(cm, tar)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_eval, losses.val, losses.avg)
                for m in metrics:
                    desc += " {} {:.4f} ({:.4f})".format(m.__name__, m.val, m.avg)

                pb.set_description(desc)
                self.logger.dump(desc)

                if self.tb_on:
                    self.tb_writer.add_image("Eval/t1", t1[0], self.eval_step)
                    self.tb_writer.add_image("Eval/t2", t2[0], self.eval_step)
                    self.tb_writer.add_image("Eval/labels", quantize(tar), self.eval_step, dataformats='HW')
                    prob = quantize(to_array(torch.exp(prob[0,1])))
                    self.tb_writer.add_image("Eval/prob", to_pseudo_color(prob), self.eval_step, dataformats='HWC')
                    self.tb_writer.add_image("Eval/cm", quantize(cm), self.eval_step, dataformats='HW')
                    self.eval_step += 1
                
                if self.save:
                    self.save_image(name[0], quantize(cm), epoch)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            self.tb_writer.add_scalars("Eval/metrics", {m.__name__.lower(): m.avg for m in metrics}, self.eval_step)
            self.tb_writer.flush()

        return metrics[2].avg   # F1-score