import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from helpers.env import load_config
import torchvision.datasets as dset
import torchvision.transforms as transforms
from data.data import SiameseNetworkDataset
from torch.utils.data import DataLoader,Dataset

CONFIG = load_config()

DATA_VAL_DIR = CONFIG["FOLDERS"]["DATA_VAL"]


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

"""
Lighting Siamese Network
"""
class SiameseNetwork(pl.LightningModule):
    def __init__(self,config):
        super(SiameseNetwork, self).__init__()

        self.threshold = config["threshold"]
        self.l1 = config["l1"]
        self.l2 = config["l2"]
        self.lr = config["lr"]

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, self.l1),
            nn.ReLU(inplace=True),

            nn.Linear(self.l1, self.l2),
            nn.ReLU(inplace=True),

            nn.Linear(self.l2, 5))


    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr = self.lr )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    def accuracy(self,input1,input2, labels):
         dist = (F.pairwise_distance(input1, input2) < self.threshold).type(torch.uint8)
         correct = (dist == labels.squeeze()).sum().item()
         accuracy = correct / len(labels)
         return torch.tensor(accuracy)


    def training_step(self, batch, batch_nb, criterion = ContrastiveLoss()):
        x1, x2, label,_ ,_ = batch
        out1, out2 = self.forward(x1,x2)
        loss = criterion(out1,out2, label)
        acc = self.accuracy(out1,out2,label)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_nb, criterion = ContrastiveLoss()):
        x1, x2, label,_ ,_ = val_batch
        out1, out2 = self.forward(x1,x2)
        acc = self.accuracy(out1,out2,label)
        loss = criterion(out1,out2, label)
        acc = self.accuracy(out1,out2,label)

        return {"val_loss": loss,"val_accuracy":acc}



    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def val_dataloader(self):
        folder_dataset_test = dset.ImageFolder(root=DATA_VAL_DIR)
        siamese_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

        return DataLoader(siamese_test,num_workers=6,batch_size=1)






"""
Lighting Siamese Network
"""
class SiameseNetworkV2(pl.LightningModule):
    def __init__(self,config):
        super(SiameseNetworkV2, self).__init__()

        self.threshold = config["threshold"]
        self.l1 = config["l1"]
        self.l2 = config["l2"]
        self.lr = config["lr"]

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(25600, self.l1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(self.l1, self.l2),
            nn.ReLU(inplace=True),

            nn.Linear(self.l2,5))


    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr = self.lr )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    def accuracy(self,input1,input2, labels):
         dist = (F.pairwise_distance(input1, input2) < self.threshold).type(torch.uint8)
         correct = (dist == labels.squeeze()).sum().item()
         accuracy = correct / len(labels)
         return torch.tensor(accuracy)


    def training_step(self, batch, batch_nb, criterion = ContrastiveLoss()):
        x1, x2, label,_ ,_ = batch
        out1, out2 = self.forward(x1,x2)
        loss = criterion(out1,out2, label)
        acc = self.accuracy(out1,out2,label)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_nb, criterion = ContrastiveLoss()):
        x1, x2, label,_ ,_ = val_batch
        out1, out2 = self.forward(x1,x2)
        acc = self.accuracy(out1,out2,label)
        loss = criterion(out1,out2, label)
        acc = self.accuracy(out1,out2,label)

        return {"val_loss": loss,"val_accuracy":acc}



    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def val_dataloader(self):
        folder_dataset_test = dset.ImageFolder(root=DATA_VAL_DIR)
        siamese_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

        return DataLoader(siamese_test,num_workers=6,batch_size=1)






def train_network(net, train_dataloader,iteration_step=10,train_number_epochs = 20):
    counter = []
    loss_history = []
    iteration_number= 0
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    for epoch in range(0,train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            #img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %iteration_step == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number += iteration_step
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    return counter, loss_history





def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
