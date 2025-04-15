# the accuracy (precision and recall) metrics we are focusing on reproducing
import torch
from torchvision.ops import box_iou

# get a confidence score based on probab

# class-level AP and AR calculation
def class_metrics(bbox_preds, gt_boxes, threshold):
    '''
    function to get the AP of any class at a threshold and AR of any class for n predictions
    assume bbox_preds are sorted by confidence 
    '''

    # Get the IoU of predicted boxes with ground truth boxes and isolate the ones above the threshold
    ious = box_iou(bbox_preds,gt_boxes)
    mask_ious = ious < threshold # where are the IoUs less than the threshold
    ious[mask_ious] = 0 # set these values to 0
    max_ious = torch.argmax(ious,dim=1) # get the index of the gt bounding box with which each pred has the largest IoU

    n_true_positives = torch.sum(class_ious > threshold)

    precision = n_true_positives / class_ious.shape[0] # number of true positives / number of all detected boxes

    recall = n_true_positives / n_gt_boxes # number of true positives / number of all gt boxes (assumes that some were not detected)

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