
import numpy as np
import torch
import torchvision
import subprocess
import sys
from sklearn import metrics

def conv_box(cood):
  if np.shape(cood)[0] == 0:
    print("Return : ", torch.from_numpy(np.array(cood)))
    return torch.from_numpy(np.array(cood))
  box_cood = np.hstack(((cood[:,1]-cood[:,3]/2).reshape(-1,1), (cood[:,2]-cood[:,4]/2).reshape(-1,1), (cood[:,1]+cood[:,3]/2).reshape(-1,1), (cood[:,2]+cood[:,4]/2).reshape(-1,1)))
  return torch.from_numpy(box_cood)


def precision_recall_curve(predictions, targets, target_class):
    target_class = int(target_class)

    thr = 0.5
    tp = 0
    fp = 0
    total_gt_count = 0
    tp_fp = []

    for i, (pred_label, tar_label) in enumerate(zip(predictions, targets)):
        #print("Target label : ", tar_label.shape)
        #print("Prediction label : ", pred_label)
        if pred_label is None or tar_label is None:
            continue
        #print("Prediction shape: ",pred_label.shape)
        pred_act_label = np.array(pred_label[target_class])
        tar_act_label = tar_label[tar_label[:,0]==target_class]

        if len(pred_act_label.shape) == 1 and pred_act_label.shape[0] != 0:
            pred_act_label = np.expand_dims(pred_act_label, 0)

        #print("Prediction : ", pred_act_label.shape)
        #print("Target : ", tar_act_label.shape)
        #print(tar_act_label)

        if np.shape(pred_act_label)[0] == 0 and np.shape(tar_act_label)[0] != 0:
            #print("Pred : Null array (False Negative)")
            total_gt_count += tar_act_label.shape[0]
            for row in range(tar_act_label.shape[0]):
                tp_fp.append(torch.tensor([tar_act_label[row,-1], 0]))
            continue
        elif np.shape(pred_act_label)[0] != 0 and np.shape(tar_act_label)[0] == 0:
            #print("Pred : Full array (False Positive)")
            for row in range(pred_act_label.shape[0]):
                tp_fp.append(torch.tensor([pred_act_label[row,-1], 1]))
            continue
        elif np.shape(pred_act_label)[0] == 0 and np.shape(tar_act_label)[0] == 0:
            #print("Nothing predicted wih no target")
            continue
        else:
            iou_out = torchvision.ops.box_iou(conv_box(pred_act_label), conv_box(tar_act_label))
            #print("iou : ", iou_out)
            #print(iou_out.shape)
            gt_tracker = []
            for row in range(iou_out.shape[0]):
                matched_indx = torch.max(iou_out[row, :], axis=0).indices
                matched_val = torch.max(iou_out[row, :], axis=0).values

                if matched_val.item() > thr:
                    if matched_indx.item() not in gt_tracker:
                        tp_fp.append(torch.tensor([pred_act_label[row,-1], 1]))
                        gt_tracker.append(matched_indx.item())
                    else:
                        tp_fp.append(torch.tensor([pred_act_label[row,-1], 2]))
                else:
                    tp_fp.append(torch.tensor([pred_act_label[row,-1], 2]))
            total_gt_count += tar_act_label.shape[0]

    if len(tp_fp) == 0:
        return [0], [0]
    tp_fp_tensor = torch.stack(tp_fp)
    tp_fp_tensor = torch.stack(sorted(tp_fp_tensor, key=lambda tp_fp_tensor: tp_fp_tensor[0]))
    precision = []
    recall = []
    tp = 0
    fp = 0
    fn = 0
    for i, out in enumerate(tp_fp_tensor):
        if out[1].item() == 1:
            tp += 1
        elif out[1].item() == 2:
            fp += 1
        elif out[1].item() == 0:
            fn += 10

        precision_val = tp / (i + 1)
        recall_val = tp / (total_gt_count + 0.00001)
        precision.append(precision_val)
        recall.append(recall_val)

    print(len(recall), len(precision))
    return recall, precision

def average_precision(predictions, targets, target_class):
    
    recall, precision = precision_recall_curve(predictions, targets, target_class)
    #print("done")
    if len(recall) == 1 :
      average_precision_val = 0
    else:
      average_precision_val = metrics.auc(np.array(recall), np.array(precision))
    return average_precision_val

def mean_average_precision(predictions, targets):
    # predictions: shape (N_image, (N_box, 6)); class, score, x1, y1, x2, y2
    #   inner array of prediction (N_box, 6) is be `None` if nothing predicted
    # targets: shape (N_image, (N_box, 5)); class, x1, y1, x2, y2
    classes = [0, 1, 2]
    map = 0
    for each_class in classes:
        map += average_precision(predictions, targets, each_class)
    map /= 3
    return map