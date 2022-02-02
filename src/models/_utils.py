import torch
import torch.nn as nn
import torch.nn.functional as F


class KaimingInitMixin:
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # By default use fan_in mode and leaky relu non-linearity with a=0
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


Identity = nn.Identity