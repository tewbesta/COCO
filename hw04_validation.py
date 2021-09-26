import argparse
from PIL import Image
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tvt
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import itertools
import pandas as pd
import os
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "orange")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#from resources.plotcm import plot_confusion_matrix
parser = argparse.ArgumentParser(description ='HW04 Training/Validation')
parser.add_argument('--root_path', type =str ,required= True)
parser.add_argument('--class_list', nargs ='*',type =str,required=True)
#parser.add_argument('--images_per_class',type =int,required=True)
args, args_other=parser.parse_known_args()

from torch.utils.data import DataLoader,Dataset
import numpy as np
from collections import defaultdict


class your_dataset_class(Dataset):
    def __init__(self,x,y,transformations,z):
        self.class_list = x
        self.transformations=transformations
        self.root_path = y
        self.images_per_class=500
        self.refrigerator_images = glob.glob(self.root_path + self.class_list[0] + '/' + '*.jpg')
        print(self.root_path + self.class_list[0])
        self.airplane_images = glob.glob(self.root_path + self.class_list[1] + '/' + '*.jpg')
        self.giraffe_images = glob.glob(self.root_path + self.class_list[2] + '/' + '*.jpg')
        self.cat_images = glob.glob(self.root_path + self.class_list[3] + '/' + '*.jpg')
        self.elephant_images = glob.glob(self.root_path + self.class_list[4] + '/' + '*.jpg')
        self.dog_images = glob.glob(self.root_path + self.class_list[5] + '/' + '*.jpg')
        self.train_images = glob.glob(self.root_path + self.class_list[6] + '/' + '*.jpg')
        self.horse_images = glob.glob(self.root_path + self.class_list[7] + '/' + '*.jpg')
        self.boat_images = glob.glob(self.root_path + self.class_list[8] + '/' + '*.jpg')
        self.truck_images = glob.glob(self.root_path + self.class_list[9] + '/' + '*.jpg')
        self.refrigerator_labels = 0
        self.airplane_labels = 1
        self.giraffe_labels = 2
        self.cat_labels = 3
        self.elephant_labels = 4
        self.dog_labels = 5
        self.train_labels = 6
        self.horse_labels = 7
        self.boat_labels = 8
        self.truck_labels = 9
        refrigerator_dict = {}
        airplane_dict = {}
        giraffe_dict = {}
        cat_dict = {}
        elephant_dict = {}
        dog_dict = {}
        train_dict = {}
        horse_dict = {}
        boat_dict = {}
        truck_dict = {}
        for i in range(self.images_per_class):
            refrigerator_images = Image.open(self.refrigerator_images[i])
            refrigerator_images = self.transformations(refrigerator_images)
            refrigerator_dict[refrigerator_images] = self.refrigerator_labels
        for i in range(self.images_per_class):
            airplane_images = Image.open(self.airplane_images[i])
            airplane_images = self.transformations(airplane_images)
            airplane_dict[airplane_images] = self.airplane_labels
        for i in range(self.images_per_class):
            giraffe_images = Image.open(self.giraffe_images[i])
            giraffe_images = self.transformations(giraffe_images)
            giraffe_dict[giraffe_images] = self.giraffe_labels
        for i in range(self.images_per_class):
            cat_images = Image.open(self.cat_images[i])
            cat_images = self.transformations(cat_images)
            cat_dict[cat_images] = self.cat_labels
        for i in range(self.images_per_class):
            elephant_images = Image.open(self.elephant_images[i])
            elephant_images = self.transformations(elephant_images)
            elephant_dict[elephant_images] = self.elephant_labels
        for i in range(self.images_per_class):
            dog_images = Image.open(self.dog_images[i])
            dog_images = self.transformations(dog_images)
            dog_dict[dog_images] = self.dog_labels
        for i in range(self.images_per_class):
            train_images = Image.open(self.train_images[i])
            train_images = self.transformations(train_images)
            train_dict[train_images] = self.train_labels
        for i in range(self.images_per_class):
            horse_images = Image.open(self.horse_images[i])
            horse_images = self.transformations(horse_images)
            horse_dict[horse_images] = self.horse_labels
        for i in range(self.images_per_class):
            boat_images = Image.open(self.boat_images[i])
            boat_images = self.transformations(boat_images)
            boat_dict[boat_images] = self.boat_labels
        for i in range(self.images_per_class):
            truck_images = Image.open(self.truck_images[i])
            truck_images = self.transformations(truck_images)
            truck_dict[truck_images] = self.truck_labels
        print(".............", len(self.truck_images))
        images1 = {**refrigerator_dict, **airplane_dict,  **giraffe_dict, **cat_dict, **elephant_dict,  **dog_dict,**train_dict, **horse_dict, **boat_dict,  **truck_dict}
        self.imageslist = list(images1.keys())
        self.labels = list(images1.values())
    def __getitem__(self,index):
        #dataset = datasets.ImageFolder(self.images, transform=self.transformations)
        return self.imageslist[index],self.labels[index]
    def __len__(self):
        self.len = len(self.imageslist)
        return self.len

transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
x=args.class_list
y=args.root_path
z=500
val_dataset=your_dataset_class(x,y,transform,z)
val_data_loader = torch.utils.data.DataLoader ( dataset=val_dataset ,batch_size =10 ,shuffle =True ,num_workers =0)
print(len(val_data_loader))


class TemplateNet3(nn.Module):
    def __init__(self):
        super(TemplateNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 2)  ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3)  ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 15 * 15, 1000)  ## (C)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # print((self.conv1(x)).size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        # x = self.pool(F.relu(self.conv2(x))) ## (D)
        # print(x.size())
        x = x.view(-1, 128 * 15 * 15)  ## (E)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.fc2(x)

        return x


class TemplateNet2(nn.Module):
    def __init__(self):
        super(TemplateNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)  ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3)  ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 1000)  ## (C)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # print((self.conv1(x)).size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        x = self.pool(F.relu(self.conv2(x)))  ## (D)
        # print(x.size())
        x = x.view(-1, 128 * 14 * 14)  ## (E)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x


class TemplateNet1(nn.Module):
    def __init__(self):
        super(TemplateNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)  ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3)  ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 31 * 31, 1000)  ## (C)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # print((self.conv1(x)).size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        # x = self.pool(F.relu(self.conv2(x))) ## (D)
        # print(x.size())
        x = x.view(-1, 128 * 31 * 31)  ## (E)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.fc2(x)
        return x


class nets():

    def __init__(self, net3, net2, net1):

        self.net3 = net3
        self.net2 = net2
        self.net1 = net1

        names = "refrigerator", "airplane", "giraffe", "cat", "elephant", "dog", "train", "horse", "boat", "truck"
        dtype = torch.float64
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        #device = torch.device(device)
        self.net3 = self.net3.to(device)
        self.net2 = self.net2.to(device)
        self.net1 = self.net1.to(device)

        print("working with CUDA...........", torch.cuda.is_available())

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net3.parameters(), lr=1e-3, momentum=0.9)
        epochs = 1
        count = 0
        loss1 = []
        loss2 = []
        loss3 = []
        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0.0
            count = 0
            cmt = torch.zeros(10, 10, dtype=torch.int64)
            for i, data in enumerate(val_data_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.net3(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print(running_loss)

                if (i + 1) % 500 == 0:
                    print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
                    epoch_loss += (running_loss / float(500))
                    running_loss = 0.0
                    count += 1

                stacked = torch.stack((labels, outputs.argmax(dim=1)), dim=1)
                for p in stacked:
                    tl, pl = p.tolist()
                    cmt[tl, pl] = cmt[tl, pl] + 1

            loss3.append(epoch_loss / count)
        print(cmt)
        plot_confusion_matrix(cmt, names)

        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0.0
            count = 0
            cmt = torch.zeros(10, 10, dtype=torch.int64)
            for i, data in enumerate(val_data_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.net2(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print(running_loss)

                if (i + 1) % 500 == 0:
                    print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
                    epoch_loss += (running_loss / float(500))
                    running_loss = 0.0
                    count += 1

                stacked = torch.stack((labels, outputs.argmax(dim=1)), dim=1)
                for p in stacked:
                    tl, pl = p.tolist()
                    cmt[tl, pl] = cmt[tl, pl] + 1

            loss2.append(epoch_loss / count)
        print(cmt)
        plot_confusion_matrix(cmt, names)
        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0.0
            count = 0
            cmt = torch.zeros(10, 10, dtype=torch.int64)
            for i, data in enumerate(val_data_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.net1(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print(running_loss)

                if (i + 1) % 500 == 0:
                    print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
                    epoch_loss += (running_loss / float(500))
                    running_loss = 0.0
                    count += 1

                stacked = torch.stack((labels, outputs.argmax(dim=1)), dim=1)
                for p in stacked:
                    tl, pl = p.tolist()
                    cmt[tl, pl] = cmt[tl, pl] + 1

            loss1.append(epoch_loss / count)
        print(cmt)
        plot_confusion_matrix(cmt, names)

        plt.figure()
        plt.plot(loss3, color='green', label="net3")
        plt.plot(loss2, color='orange', label="net2")
        plt.plot(loss1, '-b', label="net1")
        plt.legend(loc="upper left")
        plt.show()


dtype = torch.float64

net3 = TemplateNet3()
net2 = TemplateNet2()
net1 = TemplateNet1()

net3.load_state_dict(torch.load("net3.pth"))
net2.load_state_dict(torch.load("net2.pth"))
net1.load_state_dict(torch.load("net1.pth"))
x = nets(net3, net2, net1)


