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
        # featmap_sizes = [feat.shape[-2:] for feat in feats]
        # anchors = self.detection_heads[0].get_anchors(featmap_sizes)

        cls_outs, reg_outs = [], []
        soft_mask_outs = []

        sparse_cls_feats_outs, sparse_reg_feats_outs = [], []
        dense_cls_feats_outs, dense_reg_feats_outs = [], []

        for head, feat in zip(self.detection_heads, feats):
            (
                cls_out,
                reg_out,
                soft_mask,
                sparse_cls_feats,
                sparse_reg_feats,
                dense_cls_feats,
                dense_reg_feats,
            ) = head(feat, stage)

            cls_outs.append(cls_out)
            reg_outs.append(reg_out)
            soft_mask_outs.append(soft_mask)

            sparse_cls_feats_outs.append(sparse_cls_feats)
            sparse_reg_feats_outs.append(sparse_reg_feats)
            dense_cls_feats_outs.append(dense_cls_feats)
            dense_reg_feats_outs.append(dense_reg_feats)

        featmap_sizes = [feat.shape[-2:] for feat in cls_outs]
        print(f"Feature map sizes: {featmap_sizes}")
        anchors = self.detection_heads[0].get_anchors(featmap_sizes)

        # for anchor in anchors:
        #     print(f"Anchor shape: {anchor.shape}")

        return (
            cls_outs,
            reg_outs,
            soft_mask_outs,
            sparse_cls_feats_outs,
            sparse_reg_feats_outs,
            dense_cls_feats_outs,
            dense_reg_feats_outs,
            feats,  # dense_feats from FPN (optional)
            anchors,  # anchors from FPN (optional)
        )
