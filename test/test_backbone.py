import os
import sys

# add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mmengine.config import Config
from mmdet.registry import MODELS

print(f"perfect")

# Load the config
cfg = Config.fromfile("configs/resnet18_fpn_feature_extractor.py")

# Disable pretrained weights in config to avoid download
# cfg.backbone.init_cfg = None
print(f"{cfg.backbone}")
print(f"{cfg.neck}")


# Build individually
backbone = MODELS.build(cfg.backbone)
neck = MODELS.build(cfg.neck)

# Use as usual
backbone.eval()
neck.eval()
print(backbone)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)
neck.to(device)


# Dummy input
dummy_input = torch.randn(1, 3, 800, 1333).to(device)

# Forward pass to extract features
with torch.no_grad():
    c_feats = backbone(dummy_input)
    p_feats = neck(c_feats)

print(f"{type(p_feats) = }")
# Print FPN outputs
for i, f in enumerate(p_feats):
    print(f"P{i+3}: {f.shape}")
