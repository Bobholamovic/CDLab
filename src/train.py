#!/usr/bin/env python3

import os
import shutil
import random
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from core.misc import R
from core.config import parse_args
    

def main():
    # Set random seed
    RNG_SEED = 1
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    torch.cuda.manual_seed(RNG_SEED)

    cudnn.deterministic = True
    cudnn.benchmark = False

    # Parse commandline arguments
    def parser_configurator(parser):
        parser.add_argument('--crop_size', type=int, default=256, metavar='P', 
                            help="patch size (default: %(default)s)")
        parser.add_argument('--tb_on', action='store_true')
        parser.add_argument('--tb_intvl', type=int, default=100)
        parser.add_argument('--suffix_off', action='store_true')
        parser.add_argument('--lr_mode', type=str, default='const')
        parser.add_argument('--step', type=int, default=200)
        parser.add_argument('--save_on', action='store_true')
        parser.add_argument('--out_dir', default='')
        parser.add_argument('--val_iters', type=int, default=16)

        return parser
    args = parse_args(parser_configurator)

    trainer = R['Trainer_switcher'](args)

    if trainer is not None:
        if args['exp_config']:
            # Make a copy of the config file
            cfg_path = osp.join(trainer.gpc.root, osp.basename(args['exp_config']))
            shutil.copy(args['exp_config'], cfg_path)
        try:
            trainer.run()
        except BaseException as e:
            import traceback
            # Catch ALL kinds of exceptions
            trainer.logger.fatal(traceback.format_exc())
            if args['debug_on']:
                breakpoint()
            exit(1)
    else:
        raise NotImplementedError("Cannot find an appropriate trainer.")


if __name__ == '__main__':
    main()