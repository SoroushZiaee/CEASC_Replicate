import os
import sys

# add patent to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# from models import Res18FPNCEASC
from utils import Lnorm


# Create random input
def test_ceasc_model(config_path, num_classes=10, batch_size=2, image_size=512):
    # Create random input
    x = torch.randn(batch_size, 3, image_size, image_size)

    # # Load model
    # model = Res18FPNCEASC(config_path=config_path, num_classes=num_classes)
    # model.eval()  # Set to eval mode

    # # Forward pass
    # with torch.no_grad():
    #     (
    #         cls_outs,
    #         reg_outs,
    #         cls_softs,
    #         reg_softs,
    #         sparse_cls_feats,
    #         sparse_reg_feats,
    #         dense_cls_feats,
    #         dense_reg_feats,
    #         fpn_feats,
    #     ) = model(x, stage="val")

    # # Print output shape info
    # print("🧪 Test CEASC Model Output Shapes:")
    # for i, (cls, reg, c_mask, r_mask) in enumerate(
    #     zip(cls_outs, reg_outs, cls_softs, reg_softs)
    # ):
    #     print(f"FPN Level P{i+3}:")
    #     print(f"\t- cls_out:         {cls.shape}")  # [B, num_classes, H, W]
    #     print(f"\t- reg_out:         {reg.shape}")  # [B, 4, H, W]
    #     print(f"\t- cls_soft_mask:   {c_mask.shape}")  # [B, 1, H, W]
    #     print(f"\t- reg_soft_mask:   {r_mask.shape}")  # [B, 1, H, W]")

    #     print(f"\t- sparse_cls_feat[0]: {sparse_cls_feats[i][0].shape}")  # [B, C, H, W]
    #     print(
    #         f"\t- dense_cls_feat[0]:  {dense_cls_feats[i][0].shape}"
    #     )  # [B, C, H, W]")

    l_norm = Lnorm()
    sparse_cls_feats = torch.randn(5, 4, batch_size, 256, 64, 64)
    dense_cls_feats = torch.randn(5, 4, batch_size, 256, 64, 64)
    cls_soft_mask = torch.randn(5, 4, batch_size, 1, 64, 64)
    # Calculate Lnorm loss
    loss = l_norm(sparse_cls_feats, cls_soft_mask, dense_cls_feats)
    print(f"Loss: {loss}")
    # Print loss shape info


if __name__ == "__main__":
    config_path = "configs/resnet18_fpn_feature_extractor.py"
    test_ceasc_model(config_path=config_path)
