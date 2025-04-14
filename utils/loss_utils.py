import torch
import torch.nn.functional as F


def decode_dfl_bins(reg_preds, num_bins=16):
    """
    Convert DFL output to continuous bounding box regression targets.

    Args:
        reg_preds: [B, N, 4 * num_bins]
        num_bins: int, number of bins

    Returns:
        deltas: [B, N, 4] in (tx, ty, tw, th)
    """
    B, N, _ = reg_preds.shape
    reg_preds = reg_preds.view(B, N, 4, num_bins)  # [B, N, 4, bins]
    prob = F.softmax(reg_preds, dim=-1)  # apply softmax over bins

    bin_values = torch.arange(
        num_bins, dtype=torch.float32, device=reg_preds.device
    )  # [bins]
    expected = (prob * bin_values).sum(dim=-1)  # [B, N, 4]

    return expected / (num_bins - 1)  # normalize back to [0, 1] scale


def prepare_predictions(cls_outs, reg_outs, num_classes=10, num_bins=16):
    batch_size = cls_outs[0].shape[0]
    num_anchors = 6

    # Process each FPN level
    all_cls_preds = []
    all_reg_preds = []

    for cls_out, reg_out in zip(cls_outs, reg_outs):
        # Get dimensions
        B, C, H, W = cls_out.shape

        # For classification: [B, A*num_classes, H, W] -> [B, H*W*A, num_classes]
        # First reshape to [B, A, num_classes, H, W]
        reshaped_cls = cls_out.view(B, num_anchors, num_classes, H, W)
        # Then permute to [B, H, W, A, num_classes]
        permuted_cls = reshaped_cls.permute(0, 3, 4, 1, 2)
        # Finally reshape to [B, H*W*A, num_classes]
        flat_cls = permuted_cls.reshape(B, H * W * num_anchors, num_classes)
        all_cls_preds.append(flat_cls)

        # For regression: [B, A*4*num_bins, H, W] -> [B, H*W*A, 4*num_bins]
        # First reshape to [B, A, 4*num_bins, H, W]
        reshaped_reg = reg_out.view(B, num_anchors, 4 * num_bins, H, W)
        # Then permute to [B, H, W, A, 4*num_bins]
        permuted_reg = reshaped_reg.permute(0, 3, 4, 1, 2)
        # Finally reshape to [B, H*W*A, 4*num_bins]
        flat_reg = permuted_reg.reshape(B, H * W * num_anchors, 4 * num_bins)
        all_reg_preds.append(flat_reg)

    # Concatenate across FPN levels
    cls_preds = torch.cat(all_cls_preds, dim=1)  # [B, N_total*A, num_classes]
    reg_preds = torch.cat(all_reg_preds, dim=1)  # [B, N_total*A, 4*num_bins]

    return cls_preds, reg_preds
