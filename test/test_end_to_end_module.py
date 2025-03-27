import os
import sys

# add patent to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import Res18FPNCEASC


# Create random input
def test_ceasc_model(config_path, num_classes=10, batch_size=2, image_size=512):
    # Create random input
    x = torch.randn(batch_size, 3, image_size, image_size)

    # Load model
    model = Res18FPNCEASC(config_path=config_path, num_classes=num_classes)
    model.eval()  # Set to eval mode

    # Forward pass
    with torch.no_grad():
        cls_outs, reg_outs, cls_softs, reg_softs = model(x, stage="val")

    # Print output shape info
    print("Test CEASC Model Output Shapes:")
    for i, (cls, reg, c_mask, r_mask) in enumerate(
        zip(cls_outs, reg_outs, cls_softs, reg_softs)
    ):
        print(f"FPN Level P{i+3}:")
        print(f"\t\t- cls_out:       {cls.shape}")  # [B, num_classes, H, W]
        print(f"\t\t- reg_out:       {reg.shape}")  # [B, 4, H, W]
        print(f"\t\t- cls_soft_mask: {c_mask.shape}")  # [B, 1, H, W]
        print(f"\t\t- reg_soft_mask: {r_mask.shape}")  # [B, 1, H, W]")


if __name__ == "__main__":
    config_path = "configs/resnet18_fpn_feature_extractor.py"
    test_ceasc_model(config_path=config_path)
