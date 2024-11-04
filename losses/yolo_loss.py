
import torch
import torch.nn as nn

def yolo_loss(output, target,device):
    # output: [B, 8, 8, 8]
    # target: [B, 8, 8, 8]
    B = output.shape[0]
    SMOOTH = 1e-6

    output = output.to(device)
    target = target.to(device)
    iou = torch.zeros(B,8,8).to(device)

    for b in range(B):

        idx_target = target[b,0,:,:] != 0.

        xc_target , yc_target = torch.where(target[b,0,:,:] != 0.)

        if len(xc_target) != 0:
            x_target = (torch.transpose(target[b,:,idx_target,], 0, 1)[:,1] + xc_target)*16
            y_target = (torch.transpose(target[b,:,idx_target,], 0, 1)[:,2] + yc_target)*16
            w_target = torch.transpose(target[b,:,idx_target,], 0, 1)[:,3]*128
            h_target = torch.transpose(target[b,:,idx_target,], 0, 1)[:,4]*128

            x_output = (torch.transpose(output[b,:,idx_target,], 0, 1)[:,1] + xc_target)*16
            y_output = (torch.transpose(output[b,:,idx_target,], 0, 1)[:,2] + yc_target)*16
            w_output = torch.transpose(output[b,:,idx_target,], 0, 1)[:,3]*128
            h_output = torch.transpose(output[b,:,idx_target,], 0, 1)[:,4]*128

            x_right_target_corner = (x_target + w_target/2)
            x_left_target_corner = (x_target - w_target/2)
            y_top_target_corner = (y_target - h_target/2)
            y_bottom_target_corner = (y_target + h_target/2)

            x_right_output_corner = (x_output + w_output/2)
            x_left_output_corner = (x_output - w_output/2)
            y_top_output_corner = (y_output - h_output/2)
            y_bottom_output_corner = (y_output + h_output/2)

            rightmost_x = torch.min(x_right_target_corner, x_right_output_corner)
            bottomost_y = torch.min(y_bottom_target_corner, y_bottom_output_corner)
            leftmost_x = torch.max(x_left_target_corner, x_left_output_corner)
            topmost_y = torch.max(y_top_target_corner, y_top_output_corner)

            intersection_area = (rightmost_x - leftmost_x) * (bottomost_y - topmost_y)

            area_box_target = (x_right_target_corner - x_left_target_corner) * (y_bottom_target_corner - y_top_target_corner)
            area_box_output = (x_right_output_corner - x_left_output_corner) * (y_bottom_output_corner - y_top_output_corner)
            union_area = area_box_target + area_box_output - intersection_area
            iou[b, xc_target, yc_target] = (intersection_area + SMOOTH) / (union_area + SMOOTH)
    
    iou = torch.clamp(iou, 0.0, 1.0)
    location_loss_xy = torch.sum(torch.sum(target[:,0]*(torch.square(output[:,1] - target[:,1]) + torch.square(output[:,2] - target[:,2])), axis=-1), axis=-1)
    location_loss_wh = torch.sum(torch.sum(target[:,0]*(torch.square(target[:,3]**0.5-output[:,3]**0.5) + torch.square(target[:,4]**0.5-output[:,4]**0.5)),-1), axis=-1)

    box_confidence_loss = torch.sum(torch.sum(target[:,0]*torch.square(target[:,0]*iou - output[:,0]), axis=-1),-1) 
    noobj_box_confidence_loss = torch.sum(torch.sum((1-target[:,0])*torch.square(target[:,0]*iou - output[:,0]), axis=-1),-1)

    conditional_loss = torch.sum(torch.sum(target[:,0]*torch.sum(torch.square(target[:,5:] - output[:,5:]),1),-1),-1)

    lambda_coord = 5
    lambda_noobj = 0.5

    loss = lambda_coord*location_loss_xy + lambda_coord*location_loss_wh + box_confidence_loss + lambda_noobj*noobj_box_confidence_loss + conditional_loss
    loss = torch.sum(loss)
    return loss.float()