import torch
import time
import numpy as np

from models import Res18FPNCEASC  # adjust import path if needed


def compute_fps(
    config_path, input_shape=(3, 1280, 800), num_classes=10, num_iters=100, warmup=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = Res18FPNCEASC(config_path=config_path, num_classes=num_classes).to(device)
    model.eval()

    # Generate random input tensor
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Warm-up
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy_input)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            model(dummy_input)
    total_time = time.time() - start_time

    fps = num_iters / total_time
    print("=" * 30)
    print(f"Input shape: {input_shape}")
    print(f"Inference FPS: {fps:.2f}")
    print("=" * 30)
    return fps


# Example usage
if __name__ == "__main__":
    config = {
        "config_path": "configs/resnet18_fpn_feature_extractor.py"  # Replace if needed
    }
    compute_fps(config_path=config["config_path"])
