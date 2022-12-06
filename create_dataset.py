import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import os
import cv2

import glob
from tqdm import tqdm
from pathlib import Path
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

dataset_path = 'Burgmein/La secchia rapita (1)'

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

"""
620 / 5,000
Translation results
"Page border"= 0
   Erasure = 1
   Burr = 2
   "Printed Text" = 3
   "Manuscript Text" = 4
   "Pause (full or almost)" = 5
   "Single note (with at least the head)" = 6
   "Multiple Notes (with at least the head)" = 7
   "Single chord (with at least heads)" = 8
   "Multiple chords (with at least heads)" = 9
   "Accidental(s) (whole or nearly so)" = 10
   "Key(s) (whole(s) or nearly)" = 11
   "Embellishment(s) (whole(s) or nearly)" = 12
   "More categories (with at least one musical score)" = 14
   "More categories (no musical scores)" = 15
   "Other (with at least one musical score)" = 16
   "Other (without musical markings)" = 17
"""



dataset1 = []
dataset2 = []

for data_path in glob.glob(dataset_path + '/*'):
    dataset1.append(glob.glob(data_path + '/*.json'))
        
dataset1 = list(flatten( dataset1))

s = "annotazione1" 
      
for i in dataset1:
    f = open(i)
    json_data = json.load(f)
    if s in json_data:
        dataset2.append(i)
    else:
        continue

classes_relevant = [2,3,4,6,7,8,9,11,12,13,14,15,16]
classes_irrelevant = [0,1,5,10,17]
data = []

dataset_list = []

for j in dataset2:
    f = open(j)
    json_data = json.load(f)
    if json_data[s] in classes_relevant:
        data.append(j)

jList = []

for a in data:
    a1=Path(a).stem
    jList.append(a1)
    
p_list = []
for data_path in glob.glob(dataset_path + '/*'):
    if data_path.endswith('.jpg') == False and data_path.endswith('.json') == False: 
        #a = Path(data_path).stem
        p_list.append(glob.glob(data_path + '/*.png'))
        
p_list = list(flatten((p_list)))

for a in p_list:
    if Path(a).stem in jList:
        dataset_list.append(a)




print(" Length of Dataset and label ",len(dataset_list),len(classes_relevant))


#######################################################
#      Create dictionary for class indexes
#######################################################

idx_to_class = {i:j for i, j in enumerate(classes_relevant)}
class_to_idx = {value:key for key,value in idx_to_class.items()}


#######################################################
#               Define Dataset Class                  #
#######################################################


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        dim = (20,20)
        image = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-1]
        label = os.path.split(label)[0]
        #label = label.split('/')[0]
        #label.removesuffix('/*.png')
        label = class_to_idx[label]
        #print(label)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
   

def op_paths():
    
    top_paths,test_paths = dataset_list[:int(0.8*len(dataset_list))],dataset_list[int(0.8*len(dataset_list)):]
    train_paths,val_paths = top_paths[:int(0.8*len(top_paths))],top_paths[int(0.8*len(top_paths)):]
    
    return train_paths, val_paths, test_paths
#######################################################
#                  Create Dataset
#######################################################
def create_dataset():
    train_paths,val_paths,test_paths = op_paths()
    train_dataset = CustomDataset(train_paths,data_transforms)
    valid_dataset = CustomDataset(val_paths,data_transforms) #test transforms are applied
    test_dataset = CustomDataset(test_paths,data_transforms)
    
    return train_dataset, valid_dataset, test_dataset

#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################
"""
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
"""