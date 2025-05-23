from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou
from mmdet.models.losses import QualityFocalLoss, DistributionFocalLoss, GIoULoss

from utils.loss_utils import decode_dfl_bins, prepare_predictions


class Lnorm(nn.Module):
    def __init__(self):
        super(Lnorm, self).__init__()

    def forward(self, c, h, f):
        """
        Args:
            c: List[List[Tensor]] - full feature map after conv per level & branch [FPN levels][branches]
            h: List[[Tensor]] - soft masks from AMM
            f: List[List[Tensor]] - output features from CE-GN
        Returns:
            l_norm: torch.Tensor scalar, with gradients, on correct device
        """
        device = f[0][0].device  # assume all tensors are on the same device
        loss = 0.0

        for i in range(len(f)):  # FPN levels
            for j in range(len(f[i])):  # branches (usually 4)
                diff = c[i][j] * h[i] - f[i][j]
                loss += torch.mean(diff**2)

        loss = loss / (len(f) * len(f[0]))
        return loss.to(device)


class Lamm(torch.nn.Module):
    def __init__(self):
        super(Lamm, self).__init__()
        return print("Lamm init called")

    def forward(
        self, h_masks, label, im_dimx=1333, im_dimy=800
    ):  # NOTE dimension defaults set based on the paper specifications for visdrone
        l = []  # will contain the loss for each layer

        device = h_masks[0].device  # assume all tensors are on the same device

        gt_masks_raw = torch.cat(
            label, dim=0
        )  # concatenate the features along the 0th dimension, just a big tensor of mask parameters, n_objects x 4

        for i in range(len(h_masks)):  # for each layer of the FPN

            hi = h_masks[i]  # the soft mask for the ith FPN layer

            w = int(hi.shape[3])  # width of FPN layer
            h = int(hi.shape[2])  # height of FPN layer
            b = int(hi.shape[0])  # of items in batch (i.e., batch size)

            tn_pixel = (
                b * h * w
            )  # the total number of pixels in Hi and GT mask for calculating loss, n_images x height x width

            gt_reshaped = torch.zeros(
                1, 1, h, w
            )  # the reshaped version of the ground truth mask, initialized as zeros

            scale_x = (
                w / im_dimx
            )  # the scaling factors to bring the label bounding boxes to the dimensions of FPN
            scale_y = h / im_dimy

            gt_mask_scaled = torch.stack(
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
                x1 = int(torch.clamp(gt_mask_scaled[o, 0].round(), 0, w - 1))
                y1 = int(torch.clamp(gt_mask_scaled[o, 1].round(), 0, h - 1))
                x2 = int(torch.clamp(gt_mask_scaled[o, 2].round(), 0, w))
                y2 = int(torch.clamp(gt_mask_scaled[o, 3].round(), 0, h))

                if x2 <= x1 or y2 <= y1 or x1 + x2 >= w or y1 + y2 >= h:
                    continue  # Skip invalid boxes

                gt_reshaped[0, 0, y1:y2, x1:x2] = 1

            # Compute pixel-wise activation ratio difference
            pi = torch.sum(gt_reshaped) / tn_pixel
            li = ((torch.sum(hi) / tn_pixel) - pi) ** 2
            l.append(li)

        # Final AMM loss = average across FPN levels
        l_amm = sum(l) / len(l)
        return l_amm.to(device)


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


class LossQualityFocal(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, iou_targets):
        """
        Args:
            pred: [B, N, C] logits
            target: [B, N] class indices
            iou_targets: [B, N] IoU scores ∈ [0, 1]
        """
        B, N, C = pred.shape
        target_one_hot = F.one_hot(target, C).float()
        pred_sigmoid = pred.sigmoid()
        pt = target_one_hot * pred_sigmoid + (1 - target_one_hot) * (1 - pred_sigmoid)
        weights = (
            iou_targets.unsqueeze(-1) * (1 - pt) + (1 - iou_targets.unsqueeze(-1)) * pt
        ).pow(self.beta)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, reduction="none"
        )
        loss = weights * bce_loss

        num_pos = (target > 0).sum().item()
        return loss.sum() / max(num_pos, 1)


class LossDistributionFocal(nn.Module):
    def __init__(self, num_bins=16):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, pred, target, pos_mask=None):
        """
        Args:
            pred: [B, N, 4*num_bins] — raw logits
            target: [B, N, 4] — ground truth in [0, 1] scale
            pos_mask: [B, N] — positive mask
        """
        B, N, _ = pred.shape
        pred = pred.view(B, N, 4, self.num_bins)
        target_scaled = target * (self.num_bins - 1)
        left_idx = target_scaled.long().clamp(0, self.num_bins - 2)
        right_idx = left_idx + 1
        weight_right = target_scaled - left_idx.float()
        weight_left = 1.0 - weight_right

        log_probs = F.log_softmax(pred, dim=-1)  # [B, N, 4, bins]

        # Get log-probs for left and right bins
        left_logp = torch.gather(
            log_probs, dim=-1, index=left_idx.unsqueeze(-1)
        ).squeeze(-1)
        right_logp = torch.gather(
            log_probs, dim=-1, index=right_idx.unsqueeze(-1)
        ).squeeze(-1)

        loss = -(weight_left * left_logp + weight_right * right_logp)  # [B, N, 4]

        if pos_mask is not None:
            loss = loss * pos_mask.unsqueeze(-1)
            num_pos = pos_mask.sum().item() * 4
        else:
            num_pos = B * N * 4

        return loss.sum() / max(num_pos, 1)


class LossGIoU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_deltas, target_boxes, anchors, pos_mask):
        """
        Args:
            pred_deltas: [B, N, 4] ∈ (tx, ty, tw, th)
            target_boxes: [B, N, 4]
            anchors: [N, 4]
            pos_mask: [B, N]
        """
        B, N, _ = pred_deltas.shape
        anchors = anchors.unsqueeze(0).expand(B, N, 4)
        pred_boxes = self.delta2bbox(anchors, pred_deltas)

        total_loss = 0.0
        total_pos = 0

        for b in range(B):
            pos = pos_mask[b]
            if pos.sum() == 0:
                continue

            pred_b = pred_boxes[b][pos]
            target_b = target_boxes[b][pos]
            giou = generalized_box_iou(pred_b, target_b)
            loss = 1.0 - giou.diagonal()
            total_loss += loss.sum()
            total_pos += len(loss)

        return total_loss / max(total_pos, 1)

    def delta2bbox(self, anchors, deltas):
        """
        Decode (tx, ty, tw, th) into (x1, y1, x2, y2)
        """
        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * widths
        ctr_y = anchors[:, :, 1] + 0.5 * heights

        dx, dy, dw, dh = deltas.unbind(-1)
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        return torch.stack([x1, y1, x2, y2], dim=2)


class LDet(nn.Module):
    def __init__(
        self, matcher=ATSSMatcher(top_k=9), num_classes=10, num_bins=16, giou_weight=1.0
    ):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.qfl = LossQualityFocal()
        self.dfl = LossDistributionFocal(num_bins=num_bins)
        self.giou = LossGIoU()
        self.giou_weight = giou_weight

    def forward(self, cls_outs, reg_outs, anchors, targets):
        feature_shapes = [feat.shape[-2:] for feat in cls_outs]
        device = cls_outs[0].device
        batch_size = len(targets["boxes"])

        # Ensure anchors are on the correct device
        anchors = [a.to(device) for a in anchors]
        all_anchors = torch.cat(anchors, dim=0)  # [N, 4]

        # Match GT to anchors
        matched_idxs, max_ious = [], []
        for b in range(batch_size):
            boxes = targets["boxes"][b].to(device)
            m_idx, iou = self.matcher(
                anchors_per_level=anchors,
                gt_boxes=targets["boxes"][b].to(device),
                image_size=targets["orig_size"][b].to(device),
                feature_shapes=feature_shapes,
                device=device,
            )
            matched_idxs.append(m_idx)
            max_ious.append(iou)
        matched_idxs = torch.stack(matched_idxs)
        max_ious = torch.stack(max_ious)

        # === Target Assignment ===
        N = all_anchors.size(0)
        cls_targets = torch.zeros((batch_size, N), dtype=torch.long, device=device)
        iou_targets = torch.zeros((batch_size, N), dtype=torch.float, device=device)
        reg_targets = torch.zeros((batch_size, N, 4), dtype=torch.float, device=device)

        for b in range(batch_size):
            pos_mask = matched_idxs[b] >= 0
            if pos_mask.any():
                gt_inds = matched_idxs[b][pos_mask]
                labels = targets["labels"][b].to(device)
                boxes = targets["boxes"][b].to(device)

                cls_targets[b, pos_mask] = labels[gt_inds]
                reg_targets[b, pos_mask] = boxes[gt_inds]
                iou_targets[b, pos_mask] = max_ious[b][pos_mask]

        # === Prepare predictions ===
        cls_preds, reg_preds = prepare_predictions(
            cls_outs, reg_outs, self.num_classes, self.num_bins
        )
        pos_mask = matched_idxs >= 0

        # === Losses ===
        loss_qfl = self.qfl(cls_preds, cls_targets, iou_targets)
        loss_dfl = self.dfl(reg_preds, reg_targets, pos_mask)
        # clamp our dfl loss
        loss_dfl = loss_dfl.clamp(min=0, max=1.0)

        reg_deltas = decode_dfl_bins(reg_preds, self.num_bins)
        loss_giou = self.giou(reg_deltas, reg_targets, all_anchors.to(device), pos_mask)

        total_loss = loss_qfl + loss_dfl + self.giou_weight * loss_giou

        return {
            "total_loss": total_loss,
            "qfl": loss_qfl,
            "dfl": loss_dfl,
            "giou": loss_giou,
        }


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


class DetectionLoss(nn.Module):
    def __init__(self, num_bins=16, num_classes=10, num_anchors=6):
        super().__init__()
        self.qfl = QualityFocalLoss(
            use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0
        )
        self.dfl = DistributionFocalLoss(reduction="mean", loss_weight=1.0)
        self.giou = GIoULoss(reduction="mean", loss_weight=1.0)
        self.bbox_coder = GFLBBoxCoder(num_bins=num_bins)
        self.matcher = ATSSMatcher(top_k=9)
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def prepare_predictions(self, cls_outs, reg_outs):
        all_cls_preds, all_reg_preds = [], []
        for cls_out, reg_out in zip(cls_outs, reg_outs):
            B, _, H, W = cls_out.shape
            cls_out = cls_out.view(B, self.num_anchors, self.num_classes, H, W)
            cls_out = cls_out.permute(0, 3, 4, 1, 2).reshape(B, -1, self.num_classes)

            reg_out = reg_out.view(B, self.num_anchors, 4 * self.num_bins, H, W)
            reg_out = reg_out.permute(0, 3, 4, 1, 2).reshape(B, -1, 4 * self.num_bins)

            all_cls_preds.append(cls_out)
            all_reg_preds.append(reg_out)

        return torch.cat(all_cls_preds, dim=1), torch.cat(all_reg_preds, dim=1)

    def forward(self, cls_outs, reg_outs, anchors, targets, device):
        feature_shapes = [feat.shape[-2:] for feat in cls_outs]
        cls_preds, reg_preds = self.prepare_predictions(cls_outs, reg_outs)
        all_anchors = torch.cat(anchors, dim=0).to(device)  # [N_total, 4]
        batch_size = cls_preds.size(0)
        num_total_anchors = all_anchors.size(0)

        matched_idxs, max_ious = [], []
        for i in range(batch_size):
            matched_idx, ious = self.matcher(
                anchors_per_level=anchors,
                gt_boxes=targets["boxes"][i].to(device),
                image_size=targets["orig_size"][i].to(device),
                feature_shapes=feature_shapes,
                device=device,
            )
            matched_idxs.append(matched_idx.to(device))
            max_ious.append(ious.to(device))

        matched_idxs = torch.stack(matched_idxs, dim=0).to(device)
        max_ious = torch.stack(max_ious, dim=0).to(device)

        # === Build targets ===
        cls_targets = torch.zeros(
            (batch_size, num_total_anchors), dtype=torch.long, device=device
        )
        iou_targets = torch.zeros(
            (batch_size, num_total_anchors), dtype=torch.float, device=device
        )
        reg_targets = torch.zeros(
            (batch_size, num_total_anchors, 4), dtype=torch.float, device=device
        )

        for i in range(batch_size):
            pos_mask = matched_idxs[i] >= 0
            if pos_mask.any():
                gt_indices = matched_idxs[i][pos_mask]
                cls_targets[i, pos_mask] = targets["labels"][i][gt_indices.cpu()].to(
                    device
                )
                iou_targets[i, pos_mask] = max_ious[i][pos_mask]
                reg_targets[i, pos_mask] = targets["boxes"][i][gt_indices.cpu()].to(
                    device
                )

        # === Compute Losses ===
        total_qfl, total_dfl, total_giou = 0.0, 0.0, 0.0
        for i in range(batch_size):
            pos_mask = matched_idxs[i] >= 0
            pos_inds = pos_mask.nonzero(as_tuple=True)[0]

            if len(pos_inds) == 0:
                continue

            # QFL
            qfl_pred = cls_preds[i][pos_inds]
            qfl_labels = cls_targets[i][pos_inds]
            qfl_scores = iou_targets[i][pos_inds]
            qfl_loss = self.qfl(qfl_pred, (qfl_labels, qfl_scores))

            # DFL
            reg_pred = reg_preds[i][pos_inds]
            pos_anchors = all_anchors[pos_inds]
            pos_gt_boxes = reg_targets[i][pos_inds]
            encoded_target = self.bbox_coder.encode(pos_anchors, pos_gt_boxes)

            reg_pred_flat = reg_pred.view(-1, self.num_bins)
            reg_target_flat = encoded_target.view(-1)
            dfl_loss = self.dfl(reg_pred_flat, reg_target_flat)

            # GIoU
            pred_boxes = decode_dfl_bbox(reg_pred, pos_anchors, num_bins=self.num_bins)
            giou_loss = self.giou(pred_boxes, pos_gt_boxes)

            total_qfl += qfl_loss
            total_dfl += dfl_loss
            total_giou += giou_loss

        valid = sum((matched_idxs[i] >= 0).any().item() for i in range(batch_size))
        normalizer = max(valid, 1)

        return {
            "qfl": total_qfl / normalizer,
            "dfl": total_dfl / normalizer,
            "giou": total_giou / normalizer,
            "total_loss": (total_qfl + total_dfl + total_giou) / normalizer,
        }
