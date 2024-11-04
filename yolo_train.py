
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.yolo_dataset import load, process_labels, reconstruct_raw_labels_threshold
from models.yolo import YOLO
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from losses.yolo_loss import yolo_loss
from utils.nms import non_max_suppression_threshold
from utils.precision_recall import mean_average_precision, precision_recall_curve

def train(device):
    
    images, raw_labels = load()
    labels = process_labels(raw_labels)
    train_size = 5500
    train_batch_size = 50
    valid_batch_size = 50

    new_images = torch.as_tensor(images[:9900])
    labels_re = torch.as_tensor(labels[:9900])
    new_images = new_images.float()

    train_data    = new_images[:train_size].to(device)
    train_labels  = labels_re[:train_size].to(device)

    val_data     = new_images[train_size:7000].to(device)
    val_labels   = labels_re[train_size:7000].to(device)

    train_set = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_set, train_batch_size, shuffle=False )
    val_set = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_set, valid_batch_size, shuffle=False)


    print("Train : ", len(train_set))
    print("Val : ", len(val_set))

    # Setup your training
    model = YOLO()
    epochs = 80

    logger = pl_loggers.TensorBoardLogger("tb_logs", name="YOLO")
    # trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=epochs, logger=logger)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    # Plot the results
    train_N = len(model.train_losses)
    # print("Train N : ", train_N)
    train_each_size = int(train_N / epochs)
    # print("Train each size : ", train_each_size)
    epoch_loss = []

    temp = 0
    for i in range(train_N):
      temp += model.train_losses[i]
      if i!=0 and i % (train_each_size - 1) == 0:
        epoch_loss.append(temp/train_each_size)
        temp = 0

    plt.plot(epoch_loss)
    plt.xticks([k for k in range(0,epochs+1,4)])
    plt.title("Training loss vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    val_N = len(model.val_losses)
    # print("Val N : ", val_N)
    val_each_size = int(val_N / epochs)
    # print("Train each size : ", train_each_size)
    epoch_loss = []

    temp = 0
    for i in range(val_N):
      temp += model.val_losses[i]
      if i!=0 and i % (val_each_size - 1) == 0:
        # print(i)
        epoch_loss.append(temp/val_each_size)
        temp = 0

    plt.plot(epoch_loss)
    plt.xticks([k for k in range(0,epochs+1,4)])
    plt.title("Validation loss vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    map_epoch = []
    for j in range(len(model.pred_epoch)):
      preds = model.pred_epoch[j]
      # print(preds[0][0])
      mappp = 0
      for i in range(len(preds)):  
        nms_res = non_max_suppression_threshold(preds[i], nms_thres=0.5)
        mappp += mean_average_precision(nms_res, reconstruct_raw_labels_threshold(preds[i].cpu().detach().numpy(), threshold=0.5))
      map_epoch.append(mappp/len(preds))
    # print(map_epoch)
    

    plt.plot(map_epoch)
    plt.title("MAP over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Map")
    plt.show()

    pr_car, rec_car = precision_recall_curve(nms_res[:], raw_labels[:], 2)
 

    pr_traffic_light, rec_traffic_light = precision_recall_curve(nms_res[:], raw_labels[:], 0)
    pr_pedestrian, rec_pedestrian = precision_recall_curve(nms_res[:], raw_labels[:], 1)

    fig = plt.figure(figsize =(11, 9))
    plt.title("Precision vs. Recall", size=20)
    plt.xlabel('Recall', size=15)
    plt.ylabel('Precision', size =15)
    plt.plot(pr_car, rec_car, label ="Car (Class 0)", color="green")
    plt.plot(pr_pedestrian, rec_pedestrian, label = "Pedestrian (Class 2)", color="red")
    plt.plot(pr_traffic_light, rec_traffic_light, label = "Traffic light (Class 1)", color="blue")
    plt.legend(loc="upper right")
    plt.show()

    torch.save(model.state_dict(), "yolo.pth")


if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    seed = 17
    torch.manual_seed(seed)
    train(device)
  