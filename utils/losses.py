import torch
import torch.nn as nn


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

    def forward(self, h, label):
        l = []  # will contain the loss for each layer
        for i in range(
            len(label)
        ):  # for the ground truth mask of each layer of the FPN
            pi = sum(label[i] > 0) / (
                label[i].shape[0] * label[i].shape[2] * label[i].shape[3]
            )  # ratio of pixels containing classified objects to total pixels in GT - now works with multiple batches by just including them in the calculation
            li = (
                (sum(h[i] > 0) / (h[i].shape[0] * h[i].shape[2] * h[i].shape[3])) - pi
            ) ** 2  # difference in ratio between the label ratio and data ratio
            l.append(li)
        l_amm = sum(l) / len(l)
        return l_amm


class Ldet(nn.Module):
    def __init__(self):
        super(Ldet, self).__init__()

    def forward(self, c, h, f):
        """
        c: we convolve the full input feature map to calculate the c
        h: our soft mask output from the AMM layers will apply to our current loss
        f: the output features of our CE-GN layers
        """
        return 0


class QualityFocalLoss(nn.Module):
    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred_logits, target_scores, target_labels):
        B, C, H, W = pred_logits.shape
        pred_logits = pred_logits.permute(0, 2, 3, 1).reshape(-1, C)
        target_scores = target_scores.view(-1)
        target_labels = target_labels.view(-1)

        one_hot_targets = torch.zeros_like(pred_logits)
        fg_mask = target_labels > 0
        class_indices = target_labels[fg_mask] - 1
        one_hot_targets[fg_mask, class_indices] = target_scores[fg_mask]

        loss = F.binary_cross_entropy_with_logits(pred_logits, one_hot_targets, reduction="none")
        pt = torch.sigmoid(pred_logits)
        focal_weight = (one_hot_targets - pt).abs().pow(self.beta)
        loss = loss * focal_weight

        return loss.sum() / (fg_mask.sum() + 1e-6)
