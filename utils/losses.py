from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss


class Lnorm(nn.Module):
    def __init__(self):
        super(Lnorm, self).__init__()

    def forward(self, c, h, f):
        """
        Args:
            c: List[List[Tensor]] - full feature map after conv per level & branch [FPN levels][branches]
            h: List[List[Tensor]] - soft masks from AMM
            f: List[List[Tensor]] - output features from CE-GN
        Returns:
            l_norm: torch.Tensor scalar, with gradients, on correct device
        """
        device = f[0][0].device  # assume all tensors are on the same device
        loss = 0.0

        for i in range(len(f)):  # FPN levels
            for j in range(len(f[i])):  # branches (usually 4)
                diff = c[i][j] * h[i][j] - f[i][j]
                loss += torch.mean(diff**2)

        loss = loss / (len(f) * len(f[0]))
        return loss.to(device)


class Lamm(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Lamm).__init__(*args, **kwargs)

    def forward(
        self, h, label, im_dimx=800, im_dimy=1333
    ):  # NOTE dimension defaults set based on the paper specifications for visdrone
        l = []  # will contain the loss for each layer

        gt_masks_raw = torch.cat(
            label, dim=0
        )  # concatenate the features along the 0th dimension, just a big tensor of mask parameters

        for i in range(len(h)):  # for each layer of the FPN

            hi = h[i]  # the soft mask for the ith FPN layer

            tn_pixel = (
                hi.shape[0] * hi.shape[2] * hi.shape[3]
            )  # the total number of pixels in Hi and GT mask for calculating loss

            gt_reshaped = torch.zeros_like(
                hi
            )  # the reshaped version of the ground truth mask, initialized as zeros

            scale_x = (
                hi.shape[3] / im_dimx
            )  # the scaling factors to bring the label bounding boxes to the dimensions of FPN
            scale_y = hi.shape[2] / im_dimy

            for n in len(label):  # for each image in the batch
                # get new gt mask scaled to the dimensions of Hi
                gt_mask_scaled = torch.cat(
                    (
                        gt_masks_raw[:, 0] * scale_x,
                        gt_masks_raw[:, 1] * scale_y,
                        gt_masks_raw[:, 2] * scale_x,
                        gt_masks_raw[:, 3] * scale_y,
                    ),
                    dim=1,
                )

                # fill the bounding boxes into an empty mask with the shape of Hi -- needs to be done iteratively for each object in the image
                for o in range(gt_mask_scaled.shape[0]):
                    x1 = int(torch.clamp(gt_mask_scaled[o, 0].round(), 0, W - 1))
                    y1 = int(torch.clamp(gt_mask_scaled[o, 1].round(), 0, H - 1))
                    x2 = int(torch.clamp(gt_mask_scaled[o, 2].round(), 0, W))
                    y2 = int(torch.clamp(gt_mask_scaled[o, 3].round(), 0, H))

                    if x2 <= x1 or y2 <= y1:
                        continue  # Skip invalid boxes

                    gt_reshaped[n, 0, y1:y2, x1:x2] = 1

            # Compute pixel-wise activation ratio difference
            pi = torch.sum(gt_reshaped) / tn_pixel
            li = ((torch.sum(hi) / tn_pixel) - pi) ** 2
            l.append(li)

        # Final AMM loss = average across FPN levels
        l_amm = sum(l) / len(l)
        return l_amm


class ATSSMatcher:
    def __init__(self, top_k=9):
        self.top_k = top_k  # number of anchors to select per level

    def __call__(self, anchors_per_level, gt_boxes, device=None):
        """
        anchors_per_level: List[Tensor[N_i, 4]] in (x1, y1, x2, y2) format
        gt_boxes: Tensor[M, 4]
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


class DFLLoss(nn.Module):
    def __init__(self, num_bins: int = 16):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer(
            "bin_centers", torch.arange(num_bins, dtype=torch.float32) + 0.5
        )

    def forward(self, pred, target):
        """
        pred: [N, 4, num_bins] (logits)
        target: [N, 4] (continuous regression targets in range [0, num_bins])
        """
        N = pred.size(0)
        device = pred.device
        bin_centers = self.bin_centers.to(device)  # [num_bins]

        pred = pred.view(-1, self.num_bins)  # [N*4, num_bins]
        target = target.view(-1)  # [N*4]

        left_idx = torch.floor(target).long().clamp(0, self.num_bins - 1)
        right_idx = (left_idx + 1).clamp(0, self.num_bins - 1)
        weight_right = target - left_idx.float()
        weight_left = 1.0 - weight_right

        loss = (
            F.cross_entropy(pred, left_idx, reduction="none") * weight_left
            + F.cross_entropy(pred, right_idx, reduction="none") * weight_right
        )
        return loss.mean()

    def decode(self, pred_logits):
        """
        Convert predicted logits [N, 4, num_bins] → expected value → [N, 4]
        """
        probs = F.softmax(pred_logits, dim=-1)  # [N, 4, num_bins]
        bin_centers = self.bin_centers.to(probs.device)
        return (probs * bin_centers).sum(dim=-1)  # [N, 4]


class LDetection(nn.Module):
    def __init__(self, num_classes: int, num_bins: int = 16, top_k: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.matcher = ATSSMatcher(top_k=top_k)
        self.dfl_loss = DFLLoss(num_bins=self.num_bins)

    def forward(self, cls_outs, reg_outs, anchors_per_level, targets):
        batch_size = cls_outs[0].shape[0]
        device = cls_outs[0].device

        print(f"\n🔍 [Device Check] Forward Pass - LDetection")
        print(f" - cls_outs[0] device: {cls_outs[0].device}")
        print(f" - reg_outs[0] device: {reg_outs[0].device}")
        print(f" - anchors_per_level[0] device: {anchors_per_level[0].device}")
        print(f" - targets['boxes'][0] device: {targets['boxes'][0].device}")
        print(f" - targets['labels'][0] device: {targets['labels'][0].device}")

        # Flatten anchors across levels [List[N_i, 4]] -> [N_total, 4]
        anchors = torch.cat(anchors_per_level, dim=0)
        print(f" - Flattened anchors device: {anchors.device}")

        total_cls_loss = 0
        total_reg_loss = 0

        for i in range(batch_size):
            print(f"\n  ➤ Image {i}:")
            print(f"   - targets['boxes'][{i}] device: {targets['boxes'][i].device}")
            print(f"   - targets['labels'][{i}] device: {targets['labels'][i].device}")

            cls_preds = self._flatten_cls_preds(cls_outs, i)
            reg_preds = self._flatten_reg_preds(reg_outs, i)

            print(f"   - cls_preds device: {cls_preds.device}")
            print(f"   - reg_preds device: {reg_preds.device}")

            matched_idxs, _ = self.matcher(
                anchors_per_level, targets["boxes"][i], device=device
            )
            print(f"   - matched_idxs device: {matched_idxs.device}")

            # move gt_boxes and gt_labels to the same device as cls_preds
            gt_boxes = targets["boxes"][i].to(device)
            gt_labels = targets["labels"][i].to(device)

            labels, label_weights, bbox_targets = self._build_targets(
                matched_idxs, anchors, gt_labels, gt_boxes, device
            )

            print(f"   - labels device: {labels.device}")
            print(f"   - label_weights device: {label_weights.device}")
            print(f"   - bbox_targets device: {bbox_targets.device}")

            # Classification loss (QFL)
            print(
                f"[DEBUG] max label = {labels.max()}, num_classes = {self.num_classes}"
            )
            print(f"[DEBUG] unique labels = {labels.unique()}")

            qfl_targets = torch.zeros_like(cls_preds)
            pos_inds = (matched_idxs >= 0).nonzero(as_tuple=True)[0]
            if pos_inds.numel() > 0:
                with torch.no_grad():
                    ious = box_iou(anchors[pos_inds], bbox_targets[pos_inds]).diag()
                    qfl_targets[pos_inds, labels[pos_inds]] = ious.clamp(min=0)

            cls_loss = F.binary_cross_entropy_with_logits(
                cls_preds, qfl_targets, reduction="none"
            )
            cls_loss = cls_loss.sum(dim=1) * label_weights
            total_cls_loss += cls_loss.mean()

            # Regression loss (GIoU)
            if pos_inds.numel() > 0:
                pred_boxes = self.delta2bbox(anchors[pos_inds], reg_preds[pos_inds])
                reg_loss = generalized_box_iou_loss(
                    pred_boxes, bbox_targets[pos_inds], reduction="mean"
                )
                total_reg_loss += reg_loss
            else:
                total_reg_loss += 0.0

        return {
            "loss_cls": total_cls_loss / batch_size,
            "loss_reg": total_reg_loss / batch_size,
            "loss_total": (total_cls_loss + total_reg_loss) / batch_size,
        }

    def _flatten_cls_preds(self, cls_outs, index):
        # Flatten all FPN level outputs to [N_total, num_classes]
        return torch.cat(
            [
                feat[index].permute(1, 2, 0).reshape(-1, self.num_classes)
                for feat in cls_outs
            ],
            dim=0,
        )

    def _flatten_reg_preds(self, reg_outs, index):

        # Flatten all FPN level outputs to [N_total, 4]
        return torch.cat(
            [
                feat[index].permute(1, 2, 0).reshape(-1, 4, self.num_bins)
                for feat in reg_outs
            ],
            dim=0,
        )

    def _build_targets(self, matched_idxs, anchors, gt_labels, gt_boxes, device):
        num_anchors = anchors.size(0)

        labels = torch.zeros((num_anchors,), dtype=torch.long, device=device)
        label_weights = torch.zeros((num_anchors,), dtype=torch.float32, device=device)
        bbox_targets = torch.zeros_like(anchors)

        pos_mask = matched_idxs >= 0
        pos_inds = pos_mask.nonzero(as_tuple=True)[0]

        if pos_inds.numel() > 0:
            matched_gt_boxes = gt_boxes[matched_idxs[pos_inds]]
            matched_gt_labels = gt_labels[matched_idxs[pos_inds]]
            labels[pos_inds] = matched_gt_labels
            label_weights[pos_inds] = 1.0
            bbox_targets[pos_inds] = matched_gt_boxes

        return labels, label_weights, bbox_targets

    def delta2bbox(self, anchors, deltas):
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        return torch.stack([x1, y1, x2, y2], dim=1)
