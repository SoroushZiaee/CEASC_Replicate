# the accuracy (precision and recall) metrics we are focusing on reproducing
import torch
from torchvision.ops import box_iou

# class-level AP and AR calculation
def class_metrics(class_ious, threshold, n_gt_boxes):
    '''
    function to get the AP of any class at a threshold and AR of any class for n predictions
    '''
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