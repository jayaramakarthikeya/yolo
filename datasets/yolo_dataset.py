
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import traceback
from utils.plot_utils import plot_bboxes

def load_data(file_name):
    return np.load(file_name, allow_pickle=True, encoding='latin1')['arr_0']

def load():
    try:
        images = load_data('images.npz')
        raw_labels = load_data('labels.npz')
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("load failed, maybe manual download")

    return images, raw_labels

# This function should compute the 8X8X8 label from the raw labels for the corresponding image.
def process_labels(raw_labels):
    # raw_lables: shape (N_image, (N_box, 5))
    img_width = 128
    img_height = 128
    grid_size = 16
    labels = []
    # Loop through each bounding box in raw_labels
    for i, one_raw_label in enumerate(raw_labels):
        label = np.zeros((8,8,8))

        centres = np.zeros((one_raw_label.shape[0], 7))
        grid_centres = np.zeros((one_raw_label.shape[0], 2))
        one_hot_enc = np.zeros((one_raw_label.shape[0], 3))

        centres[:,0] = (one_raw_label[:,1] + one_raw_label[:,3]) / 2 # x_mid = (x1 + x2) / 2
        centres[:,1] = (one_raw_label[:,2] + one_raw_label[:,4]) / 2 # y_mid = (y1 + y2) / 2

        centres[:,2:4] = (centres[:,:2] % 16) / 16
        centres[:,4] = (one_raw_label[:,3] - one_raw_label[:,1]) / img_width
        centres[:,5] = (one_raw_label[:,4] - one_raw_label[:,2]) / img_height

        obj_class = one_raw_label[:,0].astype(int).reshape(-1,1)               # Object Classes

        one_hot_enc = np.where(obj_class==0, np.array([1,0,0]), one_hot_enc)      # Encoding
        one_hot_enc = np.where(obj_class==1, np.array([0,1,0]), one_hot_enc)
        one_hot_enc = np.where(obj_class==2, np.array([0,0,1]), one_hot_enc)

        grid_centres = centres[:,:2] // 16                             # Grid index

        final_centres = np.hstack((centres, grid_centres))

        obj_ind = final_centres[:,7:9].astype(int)          
        label[0, obj_ind[:,0], obj_ind[:,1]] = 1                  # Populating the label array
        label[1, obj_ind[:,0], obj_ind[:,1]] = final_centres[:, 2]
        label[2, obj_ind[:,0], obj_ind[:,1]] = final_centres[:, 3]
        label[3, obj_ind[:,0], obj_ind[:,1]] = final_centres[:, 4]
        label[4, obj_ind[:,0], obj_ind[:,1]] = final_centres[:, 5]
        label[5, obj_ind[:,0], obj_ind[:,1]] = one_hot_enc[:,0]
        label[6, obj_ind[:,0], obj_ind[:,1]] = one_hot_enc[:,1]
        label[7, obj_ind[:,0], obj_ind[:,1]] = one_hot_enc[:,2]

        labels.append(label)
    labels = np.array(labels)
    return labels

# This function should perform the inverse operation of process_labels().
def reconstruct_raw_labels(labels, img_dim=128, include_score=False):
    # labels: [N_image, 8, 8, 8]
    raw_labels = []
    for lbl_idx in range(0, len(labels)):
      vect = np.empty(0)
      one_raw_label = np.empty(0)
      # find in channel 1 which elements have Pr(objectness) == 1. If it does have 1, then return the indices
      # print("Here : ", labels)
    
      r,c = np.where(labels[lbl_idx][0] > 0.0)
          

      for k in range(0,len(r)):
        # Put entire row of channels in a single vector
        vect_channels = np.array(labels[lbl_idx][:, r[k], c[k]])

        # convert one-hot to integer
        onehot_to_int = np.argmax(vect_channels[len(vect_channels)-3: len(vect_channels)])
        onehot_val = np.max(vect_channels[len(vect_channels)-3: len(vect_channels)])
       

        # The channels 2, 3 -  x ,  y  coordinates represent the center of the box relative to the bounds of the grid cell
        # 4, 5 - w , h  is relative to the image width and height.
        xc = (vect_channels[1]+r[k])*16
        yc = (vect_channels[2]+c[k])*16
        w = vect_channels[3]*img_dim
        h = vect_channels[4]*img_dim

        x_ul = xc - w/2
        y_ul = yc - h/2
        x_br = xc + w/2
        y_br = yc + h/2

        # Generate one raw label
        if include_score:
          one_raw_label = np.append(one_raw_label, np.array([onehot_to_int, x_ul, y_ul, x_br, y_br, labels[lbl_idx,0, r[k], c[k]]]))
        else:
          one_raw_label = np.append(one_raw_label, np.array([onehot_to_int, x_ul, y_ul, x_br, y_br]))
     
      if include_score:
        one_raw_label = np.reshape(one_raw_label, (len(r), 6))
      else:
        one_raw_label = np.reshape(one_raw_label, (len(r), 5))
        
      # Append that one raw label to list of raw labels and return
      raw_labels.append(one_raw_label)

    return raw_labels

def reconstruct_raw_labels_threshold(labels, img_dim=128, include_score=False, threshold=0.0):
    # labels: [N_image, 8, 8, 8]
    raw_labels = []
    for lbl_idx in range(0, len(labels)):
      vect = np.empty(0)
      one_raw_label = np.empty(0)
      # find in channel 1 which elements have Pr(objectness) == 1. If it does have 1, then return the indices
      # print("Here : ", labels)
    
      r,c = np.where(labels[lbl_idx][0] > threshold)
          

      for k in range(0,len(r)):
        # Put entire row of channels in a single vector
        vect_channels = np.array(labels[lbl_idx][:, r[k], c[k]])

        # convert one-hot to integer
        onehot_to_int = np.argmax(vect_channels[len(vect_channels)-3: len(vect_channels)])
        onehot_val = np.max(vect_channels[len(vect_channels)-3: len(vect_channels)])
       

        # The channels 2, 3 -  x ,  y  coordinates represent the center of the box relative to the bounds of the grid cell
        # 4, 5 - w , h  is relative to the image width and height.
        xc = (vect_channels[1]+r[k])*16
        yc = (vect_channels[2]+c[k])*16
        w = vect_channels[3]*img_dim
        h = vect_channels[4]*img_dim

        x_ul = xc - w/2
        y_ul = yc - h/2
        x_br = xc + w/2
        y_br = yc + h/2

        # Generate one raw label
        if include_score:
          one_raw_label = np.append(one_raw_label, np.array([onehot_to_int, x_ul, y_ul, x_br, y_br, labels[lbl_idx,0, r[k], c[k]]]))
        else:
          one_raw_label = np.append(one_raw_label, np.array([onehot_to_int, x_ul, y_ul, x_br, y_br]))
     
      if include_score:
        one_raw_label = np.reshape(one_raw_label, (len(r), 6))
      else:
        one_raw_label = np.reshape(one_raw_label, (len(r), 5))
        
      # Append that one raw label to list of raw labels and return
      raw_labels.append(one_raw_label)

    return raw_labels

if __name__ == "__main__":
    images, raw_labels = load()
    labels = process_labels(raw_labels)
    plot_bboxes(images[12:14],raw_labels[12:14])
    reconstructed_raw_labels = reconstruct_raw_labels(labels)
    plot_bboxes(images[12:14],reconstructed_raw_labels[12:14])