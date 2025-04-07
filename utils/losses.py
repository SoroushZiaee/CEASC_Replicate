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

    def forward(self, h, label, im_dimx=800, im_dimy=1333): # NOTE dimension defaults set based on the paper specifications for visdrone
        l = []  # will contain the loss for each layer

        gt_masks_raw = torch.cat(label,dim=0) # concatenate the features along the 0th dimension, just a big tensor of features

        for i in range(
            len(h)
        ):  # for each layer of the FPN 
            
            hi = h[i] # the soft mask for the ith FPN layer

            tn_pixel = hi.shape[0] * hi.shape[2] * hi.shape[3] # the total number of pixels in Hi and GT mask for calculating loss

            gt_reshaped = torch.empty_like(hi) # the reshaped version of the ground truth mask

            scale_x = hi.shape[3]/im_dimx # the scaling factors to bring the label bounding boxes to the dimensions of FPN
            scale_y = hi.shape[2]/im_dimy
            
            for n in len(label): # for each image in the batch
                # get new gt mask scaled to the dimensions of Hi
                gt_mask_scaled = torch.cat((gt_masks_raw[:,0]*scale_x, 
                                            gt_masks_raw[:,1]*scale_y, 
                                            gt_masks_raw[:,2]*scale_x,
                                            gt_masks_raw[:,3]*scale_y), dim=1)
                
                # fill the bounding boxes into an empty mask with the shape of Hi -- start here, test this and see if it works
                gt_reshaped[n, 0, gt_mask_scaled[:,1]:gt_mask_scaled[:,1]+gt_mask_scaled[:,3], gt_mask_scaled[:,0]:gt_mask_scaled[:,0]+gt_mask_scaled[:,2]] = 1


            # now we can actually go ahead and calculate the loss for each layer of the FPN
            pi = sum(gt_reshaped > 0) / (tn_pixel)  # ratio of pixels containing classified objects to total pixels in GT - now works with multiple batches by just including them in the calculation
            li = (
                (sum(hi > 0) / tn_pixel) - pi
            ) ** 2  # difference in ratio between the label and Hi activation ratio
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
