import torch
import torch.nn as nn
from mmengine.config import Config
from mmdet.registry import MODELS


class ResNet18FPN(nn.Module):
    def __init__(self, config_path: str, pretrained: bool = True):
        super().__init__()

        # Load config
        cfg = Config.fromfile(config_path)

        # Disable pretrained weights if needed
        if not pretrained:
            cfg.backbone.init_cfg = None

        # Build trainable modules
        self.backbone = MODELS.build(cfg.backbone)
        self.neck = MODELS.build(cfg.neck)

        # Optional: initialize weights here if not pretrained
        # (MMDetection will already do that via init_cfg if enabled)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            List of 5 FPN feature maps [P3 to P7]
        """
        c_feats = self.backbone(x)
        p_feats = self.neck(c_feats)
        return p_feats
