
import torch
from datasets.yolo_dataset import load, process_labels, reconstruct_raw_labels, reconstruct_raw_labels_threshold
import matplotlib.pyplot as plt
import cv2
from utils.nms import low_confidence_suppression, non_max_suppression_threshold
from utils.precision_recall import precision_recall_curve
import numpy as np


def inference(device):
    model = torch.load("yolo_model.pt")
    model.eval()
    model.to(device)

    images, raw_labels = load()
    labels = process_labels(raw_labels)
    predictions = model(torch.tensor(images[:100],dtype=torch.float32).reshape(100,3,128,128).cuda())
    predictions = predictions.cpu()

    disp_img_number = 42
    """
    ORIGINAL IMAGE
    """
    plt.figure(figsize = (5,5))
    plt.title("Original image")
    plt.imshow(images[disp_img_number])


    """
    RAW LABELS
    """
    image_orig_lab = images[disp_img_number].copy()

    for i in range(0, len(raw_labels[disp_img_number])):
      x_ul = int(raw_labels[disp_img_number][i][1])
      y_ul = int(raw_labels[disp_img_number][i][2])

      x_br = int(raw_labels[disp_img_number][i][3])
      y_br = int(raw_labels[disp_img_number][i][4])

      color = (0,0,255)

    image_orig_lab = cv2.rectangle(image_orig_lab, (x_ul, y_ul), (x_br, y_br), color, 1)
    plt.figure(figsize = (5,5))
    plt.title("Image with raw label")
    plt.imshow(image_orig_lab)


    """
    RAW PREDICTIONS
    """

    predict_lab = reconstruct_raw_labels([predictions[disp_img_number].detach().numpy()])

    raw_pred_image = images[disp_img_number].copy()

    for i in range(0, len(predict_lab[0])):
      # print(predict_lab[0])
      x_ul = int(predict_lab[0][i][1])
      y_ul = int(predict_lab[0][i][2])

      x_br = int(predict_lab[0][i][3])
      y_br = int(predict_lab[0][i][4])

      color = (0,0,255)

      raw_pred_image = cv2.rectangle(raw_pred_image, (x_ul, y_ul), (x_br, y_br), color, 1)
    plt.figure(figsize = (5,5))
    plt.title("Image with Predicted label")
    plt.imshow(raw_pred_image)


    """
    LCS PREDICTIONS
    """
    lab1 = low_confidence_suppression(predictions[disp_img_number])

    lcs_label = reconstruct_raw_labels_threshold([lab1.detach().numpy()],threshold= 0.00)

    lcs_pred_image = images[disp_img_number].copy()

    for i in range(0, len(lcs_label[0])):
      # print(predict_lab[0])
      x_ul = int(lcs_label[0][i][1])
      y_ul = int(lcs_label[0][i][2])

      x_br = int(lcs_label[0][i][3])
      y_br = int(lcs_label[0][i][4])
      # print(x_ul)
      # print(y_ul)
      # print(x_br)
      # print(y_br)
      color = (0,0,255)

      lcs_pred_image = cv2.rectangle(lcs_pred_image, (x_ul, y_ul), (x_br, y_br), color, 1)
    plt.figure(figsize = (5,5))
    plt.title("Image with Predicted label LCS")
    plt.imshow(lcs_pred_image)

    """
    NMS PREDICTIONS
    """
    lab2 = low_confidence_suppression(predictions[disp_img_number])
    nms_ = non_max_suppression_threshold(torch.tensor([lab2.detach().numpy()]),nms_thres=0.0)[0]
    nms_123 = []
    for x in range(0,len(nms_)):
      if len(nms_[x]) == 0:
        continue
      if len(nms_123) == 0:
        nms_123 = nms_[x]
        continue
      nms_123 = np.vstack((nms_123,nms_[x]))

    # nms_123 = np.vstack((nms_[0],nms_[2]))
    nms_pred_image = images[disp_img_number].copy()
    # print(nms_123)
    for i in range(0, len(nms_123)):
      # print(predict_lab[0])
      x_ul = int(nms_123[i][1])
      y_ul = int(nms_123[i][2])

      x_br = int(nms_123[i][3])
      y_br = int(nms_123[i][4])
      color = (0,0,255)

      nms_pred_image = cv2.rectangle(nms_pred_image, (x_ul, y_ul), (x_br, y_br), color, 1)
    plt.figure(figsize = (5,5))
    plt.title("Image with Predicted label NMS")
    plt.imshow(nms_pred_image)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()    
    device = torch.device("cuda:0" if use_cuda else "cpu")
    inference(device)