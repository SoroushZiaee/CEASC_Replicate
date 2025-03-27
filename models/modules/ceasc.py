import torch.nn as nn
import torch
import torch.nn.functional as F

from models.modules.amm import AMM_module
from models.modules.cesc import CESC


class CEASC(nn.Module):
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
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x, stage: str = "train"):
        # compute masks
        cls_mask_hard, cls_mask_soft = self.cls_amm(x, stage)
        reg_mask_hard, reg_mask_soft = self.reg_amm(x, stage)

        # global features and pointwise conv
        global_context = F.adaptive_avg_pool2d(x, (1, 1))
        global_cls = self.global_cls(global_context)
        global_reg = self.global_reg(global_context)

        # apply 4 CESC modules
        # Fix: Chain them as in a real sequential model:
        cls_feat = x
        for cesc in self.cls_convs:
            cls_feat = cesc(cls_feat, cls_mask_hard, global_cls)

        reg_feat = x
        for cesc in self.reg_convs:
            reg_feat = cesc(reg_feat, reg_mask_hard, global_reg)

        # final prediction
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)

        # return our soft masks for loss(amm) calculation
        return cls_out, reg_out, cls_mask_soft, reg_mask_soft
