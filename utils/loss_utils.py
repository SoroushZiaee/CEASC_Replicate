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


def decode_dfl_bbox(reg_pred_logits, anchors, num_bins=16):
    B, _ = reg_pred_logits.shape
    reg_pred_logits = reg_pred_logits.view(B, 4, num_bins)
    prob = torch.softmax(reg_pred_logits, dim=2)
    bins = torch.arange(num_bins, dtype=prob.dtype, device=prob.device)
    dist = torch.sum(prob * bins, dim=2) / (num_bins - 1)

    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]

    l = dist[:, 0] * anchor_w
    t = dist[:, 1] * anchor_h
    r = dist[:, 2] * anchor_w
    b = dist[:, 3] * anchor_h

    x1 = anchors[:, 0] - l
    y1 = anchors[:, 1] - t
    x2 = anchors[:, 2] + r
    y2 = anchors[:, 3] + b

    return torch.stack([x1, y1, x2, y2], dim=1)


class GFLBBoxCoder:
    def __init__(self, num_bins=16):
        self.num_bins = num_bins

    def encode(self, anchors, gt_boxes):
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        t = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_h * (self.num_bins - 1)
        l = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_w * (self.num_bins - 1)
        b = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_h * (self.num_bins - 1)
        r = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_w * (self.num_bins - 1)
        reg_targets = torch.stack([l, t, r, b], dim=-1).clamp(0, self.num_bins - 1)
        return reg_targets


class ATSSMatcher:
    def __init__(self, top_k=9):
        self.top_k = top_k  # number of anchors to select per level

    def __call__(
        self, anchors_per_level, gt_boxes, image_size, feature_shapes, device=None
    ):
        """
        anchors_per_level: List[Tensor[N_i, 4]] in (x1, y1, x2, y2) format
        gt_boxes: Tensor[M, 4] in original image space (x1, y1, x2, y2)
        image_size: (H, W) original image size
        feature_shapes: List of (H_i, W_i) for each FPN level
        Returns:
            matched_idxs: Tensor[N_total] with GT index or -1
            max_ious: Tensor[N_total]
        """

        num_gt = gt_boxes.size(0)
        all_anchors = torch.cat(anchors_per_level, dim=0)  # [N_total, 4]
        num_anchors = all_anchors.size(0)

        if device:
            all_anchors = all_anchors.to(device)
            gt_boxes = gt_boxes.to(device)

        matched_idxs = torch.full(
            (num_anchors,), -1, dtype=torch.long, device=gt_boxes.device
        )
        max_ious = torch.zeros(num_anchors, dtype=torch.float, device=gt_boxes.device)

        # 1. Compute IoU between all anchors and GTs
        ious = ops.box_iou(all_anchors, gt_boxes)  # [N_total, M]

        # 2. Compute anchor centers
        anchor_centers = (all_anchors[:, :2] + all_anchors[:, 2:]) / 2  # [N, 2]
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # [M, 2]

        for gt_idx in range(num_gt):
            gt_box = gt_boxes[gt_idx]
            gt_center = gt_centers[gt_idx]  # [2]

            # Distance from GT center to anchor centers
            distances = torch.norm(anchor_centers - gt_center[None, :], dim=1)  # [N]

            # Pick top-k closest anchors
            topk_idxs = torch.topk(
                distances, self.top_k, largest=False
            ).indices  # [top_k]

            topk_ious = ious[topk_idxs, gt_idx]
            iou_mean = topk_ious.mean()
            iou_std = topk_ious.std()
            dynamic_thresh = iou_mean + iou_std

            # Positive = anchors with IoU >= dynamic_thresh and inside GT
            candidate_mask = ious[:, gt_idx] >= dynamic_thresh

            inside_gt = self.anchor_inside_box(all_anchors, gt_box)
            pos_mask = candidate_mask & inside_gt  # [N]

            pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
            matched_idxs[pos_indices] = gt_idx
            max_ious[pos_indices] = ious[pos_indices, gt_idx]

        return matched_idxs, max_ious

    def anchor_inside_box(self, anchors, gt_box):
        """
        Return a mask of anchors whose center is inside the GT box.
        """
        cx = (anchors[:, 0] + anchors[:, 2]) / 2
        cy = (anchors[:, 1] + anchors[:, 3]) / 2

        return (
            (cx >= gt_box[0])
            & (cx <= gt_box[2])
            & (cy >= gt_box[1])
            & (cy <= gt_box[3])
        )


def preprocess_for_metrics(
    cls_outs,
    reg_outs,
    anchors,
    targets,
    matcher,
    bbox_coder,
    num_bins,
    num_classes,
    num_anchors,
    device,
):
    """
    Prepares predictions and ground truth targets for metric computation.

    Returns:
        cls_preds_softmax: Tensor [B, A, num_classes]
        reg_preds_decoded: Tensor [B, A, 4]
        cls_targets: Tensor [B, A]
        reg_targets: Tensor [B, A, 4]
    """
    B = cls_outs[0].shape[0]
    feature_shapes = [feat.shape[-2:] for feat in cls_outs]

    # Flatten predictions
    cls_preds, reg_preds = [], []
    for cls_out, reg_out in zip(cls_outs, reg_outs):
        cls_out = (
            cls_out.view(B, num_anchors, num_classes, *cls_out.shape[-2:])
            .permute(0, 3, 4, 1, 2)
            .reshape(B, -1, num_classes)
        )
        reg_out = (
            reg_out.view(B, num_anchors, 4 * num_bins, *reg_out.shape[-2:])
            .permute(0, 3, 4, 1, 2)
            .reshape(B, -1, 4 * num_bins)
        )
        cls_preds.append(cls_out)
        reg_preds.append(reg_out)

    cls_preds = torch.cat(cls_preds, dim=1)
    reg_preds = torch.cat(reg_preds, dim=1)

    # Apply softmax over anchors for each class
    cls_preds_softmax = F.softmax(cls_preds, dim=1)

    # Flatten anchors
    all_anchors = torch.cat(anchors, dim=0).to(device)  # [A, 4]

    # Prepare targets
    cls_targets = torch.zeros((B, all_anchors.size(0)), dtype=torch.long, device=device)
    reg_targets = torch.zeros(
        (B, all_anchors.size(0), 4), dtype=torch.float, device=device
    )

    for i in range(B):
        matched_idx, _ = matcher(
            anchors_per_level=anchors,
            gt_boxes=targets["boxes"][i].to(device),
            image_size=targets["orig_size"][i].to(device),
            feature_shapes=feature_shapes,
            device=device,
        )
        pos_mask = matched_idx >= 0
        if pos_mask.any():
            gt_idx = matched_idx[pos_mask]
            cls_targets[i, pos_mask] = targets["labels"][i][gt_idx.cpu()].to(device)
            reg_targets[i, pos_mask] = targets["boxes"][i][gt_idx.cpu()].to(device)

    # Decode regression output to bounding boxes [B, A, 4]
    reg_preds_decoded = torch.stack(
        [
            decode_dfl_bbox(reg_preds[i], all_anchors, num_bins=num_bins)
            for i in range(B)
        ]
    )

    return cls_preds_softmax, reg_preds_decoded, cls_targets, reg_targets


# HOW TO USE THIS FUNCTION

# from utils.metrics_utils import preprocess_for_metrics  # if you save it in a module

# Required input variables
# cls_outs, reg_outs, anchors: from model output
# targets: a dict with 'boxes', 'labels', and 'orig_size', one per batch
# matcher: your ATSSMatcher instance
# bbox_coder: your GFLBBoxCoder instance
# device: 'cuda' or 'cpu'

# Additional hyperparameters
# num_bins = 16
# num_classes = 10
# num_anchors = 6

# # === Call preprocessing function ===
# cls_preds_softmax, reg_preds_decoded, cls_targets, reg_targets = preprocess_for_metrics(
#     cls_outs=cls_outs,
#     reg_outs=reg_outs,
#     anchors=anchors,
#     targets=targets,
#     matcher=loss_fn.matcher,           # reuse your initialized DetectionLoss instance
#     bbox_coder=loss_fn.bbox_coder,     # reuse from DetectionLoss
#     num_bins=num_bins,
#     num_classes=num_classes,
#     num_anchors=num_anchors,
#     device=device
# )
