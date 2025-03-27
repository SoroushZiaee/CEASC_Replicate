import torch.nn as nn
import torch
import torch.nn.functional as F

from models.modules.amm import AMM_module
from models.modules.cesc import CESC


class CEASC(nn.module):
    def __init__(self, in_channels, out_channels, num_classes: int = 10):
        super(CEASC, self).__init__()

        # AMM modules here:
        self.cls_amm = AMM_module(in_channels)
        self.reg_amm = AMM_module(in_channels)

        # 4*CESC module
        self.cls_convs = nn.Sequential(
            *[CESC(in_channels, in_channels) for _ in range(4)]
        )
        self.reg_convs = nn.Sequential(
            *[CESC(in_channels, in_channels) for _ in range(4)]
        )

        # Global features
        self.global_cls = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.global_reg = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Final predictino layer (sparse conv)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1, padding=1)

    def forward(self, x, stage: str = "train"):
        # compute masks
        cls_mask = self.cls_amm(x,stage)
        reg_mask = self.reg_amm(x,stage)

        # global features and pointwise conv
        global_context = F.adaptive_avg_pool2d(x, (1, 1))
        global_cls = self.global_cls(global_context)
        global_reg = self.global_reg(global_context)

        # apply 4 CESC modules
        for cesc in self.cls_convs:
            cls_feat = cesc(x, cls_mask, global_cls)
        for cesc in self.reg_convs:
            reg_feat = cesc(x, reg_mask, global_reg)

        # final prediction
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)

        return cls_out, reg_out
