import pandas as pd
from helpers.env import load_config
import os
from .structures import RawProcess, IntProcess, DataFlowProcess
import random
import torch
from PIL import Image
import PIL.ImageOps
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt



class SiameseNetworkDataset(Dataset):

    def __init__(self,imageFolderDataset,transform=None,should_invert=True,seed = None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.seed = seed


    def __getitem__(self,index):
        img0, label = self.imageFolderDataset.imgs[index]
        random.seed(self.seed)
        equal = random.randint(0,1)
        if equal:
            img1,label1 = random.choice(list(filter(lambda x: x[1]==label,self.imageFolderDataset.imgs)))
        else:
            img1,label1 = random.choice(list(filter(lambda x: x[1]!=label,self.imageFolderDataset.imgs)))
        img0 = self.process_img(img0)
        img1 = self.process_img(img1)

        return img0, img1 , torch.from_numpy(np.array([equal],dtype=np.float32)), label, label1

    def get_idx_from_folder_name(self,fname='s8'):
        cl = self.imageFolderDataset.class_to_idx[fname]
        return list(map(lambda x: x[0],list(filter(lambda x: x[1][1]==cl, list(enumerate(self.imageFolderDataset.imgs))))))
    def iter_over_all_img_dataset(self):
        for img,label in self.imageFolderDataset.imgs:
            img = self.process_img(img)
            yield img,label



    def process_img(self, img):
        if type(img)==str:
            img = Image.open(img).convert("L")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def imshow(self,img,text=None,should_save=False):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic',fontweight='bold',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def __len__(self):
        return len(self.imageFolderDataset.imgs)
