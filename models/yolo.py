
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from losses.yolo_loss import yolo_loss

class YOLO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.train_losses = []
        self.val_losses = []
        self.pred = []
        self.pred_epoch = []
        self.train_batch_size = 50
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2= nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=512)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=1024)
        self.relu6 = nn.ReLU()

        self.transposed_conv7 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=256)
        self.relu7 = nn.ReLU()

        self.transposed_conv8 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=64)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.transposed_conv7(x)))
        x = self.relu8(self.bn8(self.transposed_conv8(x)))
        x = self.conv9(x)
        x = self.sig(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        #print(images.shape)
        images = images.clone().detach().view(self.train_batch_size, 3, 128, 128)
        images = images.type(torch.float32)
        labels = labels.type(torch.float32)
        predictions = self.forward(images)
        self.pred.append(predictions)
        loss = yolo_loss(predictions, labels)
        self.train_losses.append(loss.detach().item())
        return loss

    def on_train_epoch_end(self):
        if len(self.pred) != 0:
            self.pred_epoch.append(self.pred)

        self.pred = []

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.clone().detach().view(self.train_batch_size, 3, 128, 128)
        images = images.type(torch.float32)
        labels = labels.type(torch.float32)

        predictions = self.forward(images)
        val_loss = yolo_loss(predictions, labels)
        self.val_losses.append(val_loss.detach().item())
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer