#!/usr/bin/env python3

import sys
sys.path.insert(0, '../src')

import core.factories as F
import core.builders as B
import impl.builders


if __name__ == '__main__':
    C = dict(
        lr=0.1,
        model='UNet+UNet+SNUNet',
        dataset='SVCD',
        optimizer='Adam+SGD+Adam',
        weight_decay=0.0,
        sched_on=True
    )
    model = F.model_factory(C['model'], C)
    optimizer = F.optim_factory(C['optimizer'], model, C)
    breakpoint()