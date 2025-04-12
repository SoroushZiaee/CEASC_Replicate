import os
import sys

# add patent to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import Res18FPNCEASC


def safe_shape(x):
    if isinstance(x, torch.Tensor):
        return x.shape
    elif isinstance(x, (list, tuple)):
        return [safe_shape(e) for e in x]
    return type(x)


# Create random input
def test_ceasc_model(config_path, num_classes=10, batch_size=2, image_size=512):
    # Create random input
    x = torch.randn(batch_size, 3, image_size, image_size)

    # Load model
    model = Res18FPNCEASC(config_path=config_path, num_classes=num_classes)
    model.eval()  # Set to eval mode

    # Forward pass
    with torch.no_grad():
        # Forward pass
        outputs = model(x, stage="val")
        (
            cls_outs,
            reg_outs,
            cls_soft_mask_outs,
            reg_soft_mask_outs,
            sparse_cls_feats_outs,
            sparse_reg_feats_outs,
            dense_cls_feats_outs,
            dense_reg_feats_outs,
            feats,
            anchors,
        ) = outputs

        # Print output shape info
        print("\n🔍 Output shapes from model:")
        for i in range(len(cls_outs)):
            print(f"--- FPN Level {i} ---")
            print(f"cls_outs[{i}]:              {safe_shape(cls_outs[i])}")
            print(f"reg_outs[{i}]:              {safe_shape(reg_outs[i])}")
            print(f"cls_soft_mask_outs[{i}]:    {safe_shape(cls_soft_mask_outs[i])}")
            print(f"reg_soft_mask_outs[{i}]:    {safe_shape(reg_soft_mask_outs[i])}")
            print(f"sparse_cls_feats[{i}]:      {safe_shape(sparse_cls_feats_outs[i])}")
            print(f"sparse_reg_feats[{i}]:      {safe_shape(sparse_reg_feats_outs[i])}")
            print(f"dense_cls_feats[{i}]:       {safe_shape(dense_cls_feats_outs[i])}")
            print(f"dense_reg_feats[{i}]:       {safe_shape(dense_reg_feats_outs[i])}")
            print(f"feats[{i}]:                 {safe_shape(feats[i])}")

        for i, anchor in enumerate(anchors):
            print(f"P{i+3} Anchors shape: {anchor.shape}")


if __name__ == "__main__":
    config_path = "configs/resnet18_fpn_feature_extractor.py"
    test_ceasc_model(config_path=config_path)
