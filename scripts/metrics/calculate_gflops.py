import torch
import numpy as np
from mmcv.cnn import get_model_complexity_info

from models import Res18FPNCEASC


def compute_flops_params(
    config_path, input_shape=(3, 1280, 800), num_classes=10, size_divisor=32
):
    # Adjust shape based on size_divisor
    h, w = input_shape[1], input_shape[2]
    if size_divisor > 0:
        h = int(np.ceil(h / size_divisor)) * size_divisor
        w = int(np.ceil(w / size_divisor)) * size_divisor
    input_shape = (3, h, w)

    # Initialize your model
    model = Res18FPNCEASC(config_path=config_path, num_classes=num_classes)
    model.eval()

    if hasattr(model, "forward_dummy"):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            f"FLOPs counter is not supported for {model.__class__.__name__}"
        )

    if torch.cuda.is_available():
        model.cuda()

    # Compute FLOPs and Params
    flops, params = get_model_complexity_info(model, input_shape, as_strings=True)
    print("=" * 30)
    print(f"Input shape: {input_shape}")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    print("=" * 30)


# Example usage
if __name__ == "__main__":
    config = {
        "config_path": "configs/resnet18_fpn_feature_extractor.py"
    }  # Replace this path
    compute_flops_params(config_path=config["config_path"])
