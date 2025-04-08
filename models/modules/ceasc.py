import torch.nn as nn
import torch
import torch.nn.functional as F

from models.modules.amm import AMM_module
from models.modules.cesc import CESC
from mmdet.models.task_modules.prior_generators import AnchorGenerator


class CEASC(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes: int = 10):
        super(CEASC, self).__init__()

        # Anchor generator
        # Note: The original code seems to use a custom anchor generator
        self.prior_generator = AnchorGenerator(
            strides=[4, 8, 16, 32, 64],
            ratios=[1.0],
            scales=[8],
            base_sizes=[16, 32, 64, 128, 256],  # matches your FPN levels P3–P7
        )

        # AMM modules here:
        self.cls_amm = AMM_module()
        self.reg_amm = AMM_module()

        # 4*CESC module
        self.cls_convs = nn.Sequential(
            *[CESC(in_channels, in_channels) for _ in range(4)]
        )
        self.reg_convs = nn.Sequential(
            *[CESC(in_channels, in_channels) for _ in range(4)]
        )

        # Dense layers
        self.cls_dense_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        self.reg_dense_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        # Global features
        self.global_cls = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.global_reg = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Final predictino layer (sparse conv)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)

    def get_anchors(self, featmap_sizes):
        device = next(self.parameters()).device  # auto-detect the device
        mlvl_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        return mlvl_anchors

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
        sparse_cls_feats = []  # F_{i,j}
        cls_feat = x
        for cesc in self.cls_convs:
            cls_feat = cesc(cls_feat, cls_mask_hard, global_cls)
            sparse_cls_feats.append(cls_feat)

        reg_feat = x
        sparse_reg_feats = []  # F_{i,j}
        for cesc in self.reg_convs:
            reg_feat = cesc(reg_feat, reg_mask_hard, global_reg)
            sparse_reg_feats.append(reg_feat)

        # dense layers
        cls_feat_dense = []
        cls_feat = x
        for dense_conv in self.cls_dense_convs:
            cls_feat = dense_conv(cls_feat)
            cls_feat_dense.append(cls_feat)

        reg_feat_dense = []
        reg_feat = x
        for dense_conv in self.reg_dense_convs:
            reg_feat = dense_conv(reg_feat)
            reg_feat_dense.append(reg_feat)

        # final prediction
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)

        return (
            cls_out,
            reg_out,  # final predictions
            cls_mask_soft,
            reg_mask_soft,  # for Lamm loss because sigmoid is differentiable but the hard thresholding is not
            sparse_cls_feats,
            sparse_reg_feats,  # F_{i,j}
            cls_feat_dense,
            reg_feat_dense,  # C_{i,j}
        )
