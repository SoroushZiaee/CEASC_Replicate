import torch.nn as nn
import torch
import torch.nn.functional as F

from models.backbone.gfl_v1_fpn import ResNet18FPN
from models.modules.ceasc import CEASC


class Res18FPNCEASC(nn.Module):
    def __init__(self, config_path: str, num_classes: int = 10):
        super(Res18FPNCEASC, self).__init__()

        self.backbone = ResNet18FPN(config_path=config_path, pretrained=True)
        self.detection_heads = nn.ModuleList(
            [CEASC(256, 256, num_classes=num_classes) for _ in range(5)]
        )

    def forward(self, x, stage: str = "train"):
        feats = self.backbone(x)
        cls_outs = []
        reg_outs = []
        cls_soft_mask_outs = []
        reg_soft_mask_outs = []

        for head, feat in zip(self.detection_heads, feats):
            cls_out, reg_out, cls_soft_mask, reg_soft_mask = head(feat, stage)
            cls_outs.append(cls_out)
            reg_outs.append(reg_out)
            cls_soft_mask_outs.append(cls_soft_mask)
            reg_soft_mask_outs.append(reg_soft_mask)

        return cls_outs, reg_outs, cls_soft_mask_outs, reg_soft_mask_outs
