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
    
    aps = np.array(threshold_aps)

    return aps

def get_APs(preds,gt_bs,gt_cs,n_classes):
    '''
    function to get AP50, AP75 and mAPs across classes 
    preds: Tensor[B,A,4] bounding box predictions in form x1,y1,x2,y2
    gt_bs: Tensor[B,A,4] bounding box ground truths in the form x1,y1,x2,y2
    gt_cs: Tensor[B,A,N_c] class labels for each anchor where N_c = number of classes
    n_classes: scalar, number of classes  
    '''
    
    all_aps = np.empty((n_classes,10)) # for 10 thresholds we want to look at 

    for c in range(n_classes):
        pred_bbs_list = []
        gt_bbs_list = []

        # get the bounding boxes for the correct class 
        for b in range(gt_cs.shape[0]):
            cls_idx = torch.squeeze(torch.argwhere(gt_cs[b,:,c]))
            for i in range(len(cls_idx)):
                pred_bbs_list.append(torch.squeeze(preds[b,i,:]))
                gt_bbs_list.append(torch.squeeze(gt_bs[b,i,:]))
        
        pred_bbs = torch.cat(pred_bbs_list,dim=0)
        gt_bbs = torch.cat(gt_bbs_list,dim=0)

        class_aps = class_AP(pred_bbs,gt_bbs)
        all_aps[c,:] = class_aps
    
    ap50 = np.nanmean(all_aps[:,0])
    ap75 = np.nanmean(all_aps[:,5])

    maps = np.nanmean(all_aps)

    return ap50, ap75, maps


def class_AR(bbox_preds_list, gt_boxes_list, threshold_min=0.50, threshold_max=0.95):
    '''
    computes the AR for a given class of images across all IoU thresholds 
    bbox_preds_list: List[Tensor[t],4]] where t is number of top predictions being considered, length of the list is the number of images
    gt_boxs_list: List[Tensor[A_c,4]] where A_c is the number of ground truth bounding boxes, length of the list is the number of images
    threshold_min: scalar, the minimum threshold for true positives 
    threshold_max: scalar, the maximum threshold for true positives 
    '''
    # get the range of testing thresholds 
    test_thresholds = range(threshold_min,threshold_max,0.05)
    threshold_img_ars = np.empty((int(len(bbox_preds_list)),10)) # store the recall for each image and threshold

    for i in range(len(bbox_preds_list)): # for every image that was predicted 
        bbox_preds = bbox_preds_list[i] # get the ith image 
        gt_boxes = gt_boxes_list[i]
        
        # Get the IoU of predicted boxes with ground truth boxes and isolate the ones above the threshold
        og_ious = box_iou(bbox_preds,gt_boxes)

        for t in range(10):
            thresh = test_thresholds[t]
            # Count the number of true positives 
            matched_gts = []
            true_positives = 0

            ious = og_ious.clone()
            mask_ious = ious < thresh # where are the IoUs less than the threshold
            ious[mask_ious] = 0 # set these values to 0
            max_ious = torch.argmax(ious,dim=1) # get the index of the gt bounding box with which each pred has the largest IoU
            
            for b in range(bbox_preds.shape[0]): # for the number of predicted bounding boxes 
                if ious[max_ious[b]] != 0 and max_ious[b] not in matched_gts:
                    matched_gts.append(max_ious[b])
                    true_positives += 1 # if the gt box was novelly detected by this box
            
            recall = true_positives/gt_boxes.shape[0]
            threshold_img_ars[i,t] = recall

    return np.nanmean(threshold_img_ars)
    

# overall function to make it easy to do this across multiple classes and thresholds
def get_ARs(preds,gt_bs,gt_cs,n_classes):
    '''
    function to get AR1, AR10, AR100, and AR500
    preds: Tensor[B,A,4] bounding box predictions in form x1,y1,x2,y2
    gt_bs: Tensor[B,A,4] bounding box ground truths in the form x1,y1,x2,y2
    gt_cs: Tensor[B,A,N_c] class labels for each anchor where N_c = number of classes
    n_classes: scalar, number of classes
    '''
    tops = [1,10,100,500] # the top n number of images of a class to get predictions for
    ar_store = np.empty((n_classes,int(len(tops)))) 

    for t in range(len(tops)):
        top = tops[t]
        for c in range(n_classes):
            pred_bbs_list = []
            gt_bbs_list = []

            # get the bounding boxes for the correct class and top t predictions for each image 
            for b in range(gt_cs.shape[0]):
                cls_idx = torch.squeeze(torch.argwhere(gt_cs[b,:,c]))
                
                img_pred_list = []
                for i in range(top): # only get the top t predicted bounding boxes 
                    img_pred_list.append(torch.squeeze(preds[b,cls_idx[i],:]))
                    img_pred = torch.cat(img_pred_list,dim=0)
                
                img_gt_list = []
                for i in range(len(cls_idx)): # get all the ground truth boxes 
                    img_gt_list.append(torch.squeeze(gt_bs[b,i,:]))
                    img_gt = torch.cat(img_gt_list,dim=1)
                
                
            pred_bbs_list.append(img_pred) # these lists store predicted and ground truth boxes for each image 
            gt_bbs_list.append(img_gt)

            class_top_ar = class_AR(pred_bbs_list,gt_bbs_list)
            ar_store[c,t] = class_top_ar

    ar1 = np.nanmean(ar_store[:,0])
    ar10 = np.nanmean(ar_store[:,1])
    ar100 = np.nanmean(ar_store[:,2])
    ar500 = np.nanmean(ar_store[:,3])

    return ar1, ar10, ar100, ar500







# mAP

# AP50 and AP75 

# AR1

# AR10

# AR100

# AR500