import argparse
import os.path as osp
from collections import ChainMap

import yaml


def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg or {}


def parse_configs(cfg_path, inherit=True):
    # Read and parse config files
    cfg_dir = osp.dirname(cfg_path)
    cfg_name = osp.basename(cfg_path)
    cfg_name, ext = osp.splitext(cfg_name)
    parts = cfg_name.split('_')
    cfg_path = osp.join(cfg_dir, parts[0])
    cfgs = []
    for part in parts[1:]:
        cfg_path = '_'.join([cfg_path, part])
        if osp.exists(cfg_path+ext):
            cfgs.append(read_config(cfg_path+ext))
    cfgs.reverse()
    if len(parts)>=2:
        return ChainMap(*cfgs, dict(tag=parts[1], suffix='_'.join(parts[2:])))
    else:
        return ChainMap(*cfgs)


def parse_args(parser_configurator=None):
    # Parse necessary arguments
    # Global settings
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('cmd', choices=['train', 'eval'])

    # Data
    group_data = parser.add_argument_group('data')
    group_data.add_argument('--dataset', type=str)
    group_data.add_argument('--num_workers', type=int, default=4)
    group_data.add_argument('--repeats', type=int, default=1)
    group_data.add_argument('--subset', type=str, default='val')

    # Optimizer
    group_optim = parser.add_argument_group('optimizer')
    group_optim.add_argument('--optimizer', type=str, default='Adam')
    group_optim.add_argument('--lr', type=float, default=1e-4)
    group_optim.add_argument('--weight_decay', type=float, default=1e-4)
    group_optim.add_argument('--load_optim', action='store_true')
    group_optim.add_argument('--save_optim', action='store_true')

    # Training related
    group_train = parser.add_argument_group('training related')
    group_train.add_argument('--batch_size', type=int, default=8)
    group_train.add_argument('--num_epochs', type=int)
    group_train.add_argument('--resume', type=str, default='')
    group_train.add_argument('--anew', action='store_true',
                        help="clear history and start from epoch 0 with weights updated")
    group_train.add_argument('--device', type=str, default='cpu')

    # Experiment
    group_exp = parser.add_argument_group('experiment related')
    group_exp.add_argument('--exp_dir', default='../exp/')
    group_exp.add_argument('--tag', type=str, default='')
    group_exp.add_argument('--suffix', type=str, default='')
    group_exp.add_argument('--exp_config', type=str, default='')
    group_exp.add_argument('--debug_on', action='store_true')
    group_exp.add_argument('--inherit_off', action='store_true')
    group_exp.add_argument('--log_off', action='store_true')
    group_exp.add_argument('--track_intvl', type=int, default=1)

    # Criterion
    group_critn = parser.add_argument_group('criterion related')
    group_critn.add_argument('--criterion', type=str, default='NLL')
    group_critn.add_argument('--weights', type=float, nargs='+', default=None)

    # Model
    group_model = parser.add_argument_group('model')
    group_model.add_argument('--model', type=str)

    if parser_configurator is not None:
        parser = parser_configurator(parser)

    args, unparsed = parser.parse_known_args()
    
    if osp.exists(args.exp_config):
        cfg = parse_configs(args.exp_config, not args.inherit_off)
        group_config = parser.add_argument_group('from_file')
        
        def _cfg2arg(cfg, parser, prefix=''):
            for k, v in cfg.items():
                if isinstance(v, (list, tuple)):
                    # Only apply to homogeneous lists and tuples
                    parser.add_argument('--'+prefix+k, type=type(v[0]), nargs='*', default=v)
                elif isinstance(v, dict):
                    # Recursively parse a dict
                    _cfg2arg(v, parser, prefix+k+'.')
                elif isinstance(v, bool):
                    parser.add_argument('--'+prefix+k, action='store_true', default=v)
                else:
                    parser.add_argument('--'+prefix+k, type=type(v), default=v)
        _cfg2arg(cfg, group_config, '')
        args = parser.parse_args()
    elif args.exp_config != '':
        raise FileNotFoundError
    elif len(unparsed)!=0:
        raise RuntimeError("Unrecognized arguments")

    def _arg2cfg(cfg, args):
        args = vars(args)
        for k, v in args.items():
            pos = k.find('.')
            if pos != -1:
                # Iteratively parse a dict
                dict_ = cfg
                while pos != -1:
                    dict_.setdefault(k[:pos], {})
                    dict_ = dict_[k[:pos]]
                    k = k[pos+1:]
                    pos = k.find('.')
                dict_[k] = v
            else:
                cfg[k] = v
        return cfg

    return _arg2cfg(dict(), args)