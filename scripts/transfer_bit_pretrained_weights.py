#!/usr/bin/env python

import sys
import torch


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    state_dict = torch.load(in_path)

    state_dict['state_dict'] = state_dict.pop('model_G_state_dict')
    for key in list(state_dict['state_dict'].keys()):
        if key.startswith('resnet.layer4') or key.startswith('resnet.fc'):
            state_dict['state_dict'].pop(key)
    state_dict['max_acc'] = (state_dict.pop('best_val_acc'), state_dict.pop('epoch_id'))
    state_dict['optimizer'] = state_dict.pop('optimizer_G_state_dict')
    # state_dict.pop('exp_lr_scheduler_G_state_dict')

    torch.save(state_dict, out_path)