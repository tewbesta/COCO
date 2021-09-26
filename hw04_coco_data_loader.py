import argparse
from PIL import Image
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tvt
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description ='HW04 Training/Validation')
parser.add_argument('--root_path', type =str ,required= True)
parser.add_argument('--class_list', nargs ='*',type =str,required=True)
parser.add_argument('--images_per_class',type =int,required=True)
args, args_other=parser.parse_known_args()

import os
from torch.utils.data import DataLoader,Dataset
import numpy as np
from collections import defaultdict

class your_dataset_class(Dataset):
    def __init__(self,x,y,transformations,z):
        self.class_list = x
        len_class_list=len(self.class_list)
        label_list = list(range(0, len_class_list))
        print(len_class_list)
        self.transformations=transformations
        self.root_path = y
        #self.images_per_class=z
        print(self.root_path + self.class_list[0])
        img_dict={}
        img_dict2={}
        for i in range(len_class_list):
            temp={}
            print(self.root_path + self.class_list[i])
            path=os.path.join(self.root_path,self.class_list[i])
            for j in range(z):
                temp=glob.glob(self.root_path + self.class_list[i] + '/' + '*.jpg')
                temp1 = Image.open(temp[j])
                temp1 = self.transformations(temp1)
                img_dict[temp1] = label_list[i]
                #print(len(os.listdir(path)))
            img_dict2 = {**img_dict}
        self.imageslist = list(img_dict.keys())
        self.labels = list(img_dict.values())
    def __getitem__(self,index):
        #dataset = datasets.ImageFolder(self.images, transform=self.transformations)
        return self.imageslist[index],self.labels[index]
    def __len__(self):
        self.len = len(self.imageslist)
        return self.len
x=args.class_list
y=args.root_path
z=args.images_per_class
transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset=your_dataset_class(x,y +"Train/",transform,z)
total=len(train_dataset)
print("the length is", total)
train_data_loader = torch.utils.data.DataLoader(dataset =train_dataset ,batch_size =10 ,shuffle =True ,num_workers =0)
val_dataset = your_dataset_class (x,y+"Val/", transform,z)
val_data_loader = torch.utils.data.DataLoader ( dataset=val_dataset ,batch_size =10 ,shuffle =True ,num_workers =0)
totalval=len(train_dataset)
dtype = torch.float64