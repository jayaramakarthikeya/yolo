
import numpy as np

def ioU_NMS(box1, box2): # (class, x, y, w, h, confidence)
  SMOOTH = 0.00001
  x_1 = box1[1]
  y_1 = box1[2]
  w_1 = box1[3]
  h_1 = box1[4]

  x_2 = box2[1]
  y_2 = box2[2]
  w_2 = box2[3]
  h_2 = box2[4]

  x_right_1_corner = (x_1 + w_1/2)
  x_left_1_corner = (x_1 - w_1/2)
  y_right_1_corner = (y_1 + h_1/2)
  y_left_1_corner = (y_1 - h_1/2)

  x_right_2_corner = (x_2 + w_2/2)
  x_left_2_corner =  (x_2 - w_2/2)
  y_right_2_corner = (y_2 + h_2/2)
  y_left_2_corner =  (y_2 - h_2/2)


  x_right_cood = np.min(np.hstack(((x_1 + w_1/2).reshape(-1,1), (x_2 + w_2/2).reshape(-1,1))), axis=1)        # Max of top right..returns max x, y coordinate of intersection bounding box
  y_right_cood = np.min(np.hstack(((y_1 + h_1/2).reshape(-1,1), (y_2 + h_2/2).reshape(-1,1))), axis=1)

  x_left_cood = np.max(np.hstack(((x_1 - w_1/2).reshape(-1,1), (x_2 - w_2/2).reshape(-1,1))), axis=1)         # Min of bottom left..returns min x, y coordinate of intersection bounding box
  y_left_cood = np.max(np.hstack(((y_1 - h_1/2).reshape(-1,1), (y_2 - h_2/2).reshape(-1,1))), axis=1)

  intersection_area = (x_right_cood[0] - x_left_cood[0]) * (y_right_cood[0] - y_left_cood[0])
  union_area = (x_right_1_corner - x_left_1_corner)*(y_right_1_corner - y_left_1_corner) + (x_right_2_corner - x_left_2_corner)*(y_right_2_corner - y_left_2_corner) - intersection_area
  iou = ((intersection_area + SMOOTH) / (union_area + SMOOTH))      

  return iou