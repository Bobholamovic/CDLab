#!/usr/bin/env python3

import argparse
import os.path as osp
from glob import iglob

import torch
import numpy as np
from skimage.io import imread, imsave

import core
import impl.builders
import impl.trainers
from core.misc import R
from core.config import parse_args
from core.factories import model_factory
from utils.metrics import Precision, Recall, F1Score, Accuracy
from utils.data_utils.preprocessors import Normalize
from utils.data_utils.misc import to_tensor, to_array, quantize_8bit


class WindowGenerator:
    def __init__(self, h, w, ch, cw, si=1, sj=1):
        self.h = h
        self.w = w
        self.ch = ch
        self.cw = cw
        if self.h < self.ch or self.w < self.cw:
            raise NotImplementedError
        self.si = si
        self.sj = sj
        self._i, self._j = 0, 0

    def __next__(self):
        # Column-first movement
        if self._i > self.h:
            raise StopIteration
        
        bottom = min(self._i+self.ch, self.h)
        right = min(self._j+self.cw, self.w)
        top = max(0, bottom-self.ch)
        left = max(0, right-self.cw)

        if self._j >= self.w-self.cw:
            if self._i >= self.h-self.ch:
                # Set an illegal value to enable early stopping
                self._i = self.h+1
            self._goto_next_row()
        else:
            self._j += self.sj
            if self._j > self.w:
                self._goto_next_row()

        return slice(top, bottom, 1), slice(left, right, 1)

    def __iter__(self):
        return self

    def _goto_next_row(self):
        self._i += self.si
        self._j = 0


class Preprocessor:
    def __init__(self, mu, sigma, device):
        self.norm = Normalize(np.asarray(mu), np.asarray(sigma))
        self.device = device

    def __call__(self, im):
        im = self.norm(im)
        im = to_tensor(im).unsqueeze(0).float()
        im = im.to(self.device)
        return im


class PostProcessor:
    def __init__(self, out_type, out_key=0):
        self.out_type = out_type
        self.out_key = out_key

    def __call__(self, pred):
        if not isinstance(pred, torch.Tensor):
            pred = pred[self.out_key]
        if self.out_type == 'logits':
            return to_array(torch.nn.functional.sigmoid(pred)[0,0])
        elif self.out_type == 'logits2':
            return to_array(torch.nn.functional.softmax(pred, dim=1)[0,1])
        elif self.out_type == 'dist':
            return to_array(pred[0,0])
        else:
            raise ValueError


def sw_infer(t1, t2, model, window_size, stride, prep, postp):
    h, w = t1.shape[:2]
    win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
    prob_map = np.zeros((h,w), dtype=np.float)
    cnt = np.zeros((h,w), dtype=np.float)
    with torch.no_grad():
        for rows, cols in win_gen:
            patch1, patch2 = t1[rows, cols], t2[rows, cols]
            patch1, patch2 = prep(patch1), prep(patch2)
            pred = model(patch1, patch2)
            prob = postp(pred)
            prob_map[rows,cols] += prob
            cnt[rows,cols] += 1
        prob_map /= cnt
    return prob_map


def prepare_model(args):
    model = model_factory(args['model'], args)
    ckp_dict = torch.load(args['ckp_path'])
    model.load_state_dict(ckp_dict['state_dict'])
    model.to(args['device'])
    model.eval()
    return model


def main():
    # Parse commandline arguments
    def parser_configurator(parser):
        # HACK: replace the original parser by a new one
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--exp_config', type=str, default='')
        parser.add_argument('--inherit_off', action='store_true')
        parser.add_argument('--ckp_path', type=str)
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--t1_dir', type=str, default='')
        parser.add_argument('--t2_dir', type=str, default='')
        parser.add_argument('--out_dir', type=str, default='')
        parser.add_argument('--gt_dir', type=str, default='')
        parser.add_argument('--window_size', type=int, default=256)
        parser.add_argument('--stride', type=int, default=256)
        parser.add_argument('--save_on', action='store_true')
        parser.add_argument('--mu', type=float, nargs='+', default=(0.0,0.0,0.0))
        parser.add_argument('--sigma', type=float, nargs='+', default=(255.0,255.0,255.0))
        parser.add_argument('--glob', type=str, default='*.png')
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--out_type', type=str, choices=['logits', 'logits2', 'dist'], default='logits')

        return parser
    
    args = parse_args(parser_configurator)
    
    logger = R['Logger']

    model = prepare_model(args)

    prep = Preprocessor(args['mu'], args['sigma'], args['device'])
    postp = PostProcessor(args['out_type'])

    prec, rec, f1, acc = Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), Accuracy(mode='accum')
    
    try:
        for i, path in enumerate(iglob(osp.join(args['gt_dir'], args['glob']))):
            basename = osp.basename(path)
            gt = (imread(path)>0).astype('uint8')
            t1 = imread(osp.join(args['t1_dir'], basename))
            t2 = imread(osp.join(args['t2_dir'], basename))
            prob_map = sw_infer(t1, t2, model, args['window_size'], args['stride'], prep, postp)
            cm = (prob_map>args['threshold']).astype('uint8')
            
            prec.update(cm, gt)
            rec.update(cm, gt)
            f1.update(cm, gt)
            acc.update(cm, gt)
            logger.show("{:>4} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} OA: {:.4f}".format(
                i, prec.val, rec.val, f1.val, acc.val
            ))

            if args['save_on']:
                imsave(osp.join(args['out_dir'], basename), quantize_8bit(cm), check_contrast=False)
    except BaseException as e:
        import traceback
        import sys
        import pdb
        logger.fatal(traceback.format_exc())
        pdb.post_mortem(sys.exc_info()[2])
        exit(1)


if __name__ == '__main__':
    main()