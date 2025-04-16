# the accuracy (precision and recall) metrics we are focusing on reproducing
import torch
from torchvision.ops import box_iou
import numpy as np
from scipy.integrate import trapezoid

# get a confidence score based on probab

# class-level AP calculation
def class_AP(bbox_preds, gt_boxes, threshold_min=0.50, threshold_max=0.95):
    '''
    function to get the AP of any class for thresholds from 0.50 to 0.95 in steps on 0.05
    assume bbox_preds are sorted by confidence 

    bbox_preds: Tensor[A_c,4] where A_c is number of anchors in this class, bounding boxes need to be sorted in order of descending confidence
    gt_boxs: Tensor[A_c,4]
    threshold_min: scalar, the minimum threshold for true positives 
    threshold_max: scalar, the maximum threshold for true positives 

    '''
    # get the range of testing thresholds 
    test_thresholds = range(threshold_min,threshold_max,0.05)
    threshold_aps = []

    # Get the IoU of predicted boxes with ground truth boxes and isolate the ones above the threshold
    og_ious = box_iou(bbox_preds,gt_boxes)

    for t in test_thresholds:
        # Two possibilities for each prediction and ground truth values that have already been matched
        matched_gts = []
        true_positives = np.empty(bbox_preds.shape[0])
        false_positives = np.empty(bbox_preds.shape[0])


        ious = og_ious.clone()
        mask_ious = ious < t # where are the IoUs less than the threshold
        ious[mask_ious] = 0 # set these values to 0
        max_ious = torch.argmax(ious,dim=1) # get the index of the gt bounding box with which each pred has the largest IoU
        
        for b in range(bbox_preds.shape[0]): # for the number of predicted bounding boxes 
            if ious[max_ious[b]] == 0 or max_ious[b] in matched_gts:
                false_positives[b] = 1  # if this gt box has already been detected or the best match has an IoU of 0
                true_positives[b] = 0
            elif max_ious[b] not in matched_gts:
                matched_gts.append(max_ious[b])
                false_positives[b] = 0
                true_positives[b] = 1 # if the gt box was novelly detected by this box
        
        # now get the values for precision and recall curves across predictions 
        accum_tp = np.cumsum(true_positives)
        accum_fp = np.cumsum(false_positives)

        recall = accum_tp/gt_boxes.shape[0]
        precision = accum_tp/(accum_tp+accum_fp+1e-6) # make sure to divide by some small constant to avoid dividing by zero

        auc = trapezoid(precision,recall)

        threshold_aps.append(auc)
    return threshold_aps

def get_APs(preds,gt_bs,gt_cs,n_classes):
    '''
    function to get AP50, AP75 and mAPs across classes 
    preds: Tensor[B,A,4] bounding box predictions in form x1,y1,x2,y2
    gt_bs: Tensor[B,A,4] bounding box ground truths in the form x1,y1,x2,y2
    gt_cs: Tensor[B,A,N_c] class labels for each anchor where N_c = number of classes
    n_classes: scalar, number of classes  
    '''
    for c in range(n_classes):
        for b in range(len(gt_cs.shape[0])):
            cls_idx = torch.argwhere(gt_cs[b,:,c])
            


# overall function to make it easy to do this across multiple classes and thresholds
def main():
    '''
    computes AP50, AP75 and mAP across these two levels 
    AR1, AR10, AR100, AR500
    '''

# mAP

# AP50 and AP75 

# AR1

# AR10

# AR100

# AR500