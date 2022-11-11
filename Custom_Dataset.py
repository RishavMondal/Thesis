
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import cv2

import glob
from tqdm import tqdm

#######################################################
#               Define Transforms
#######################################################
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

####################################################
#       Create data paths
####################################################

dataset_path = 'Burgmein/La secchia rapita (1)'

dataset_paths = [] #to store image paths in list
classes = [] #to store class values


for data_path in glob.glob(dataset_path + '/*'):
    if data_path.endswith('.jpg') == False and data_path.endswith('.json') == False: 
        classes.append(data_path.split('/')[-1]) 
        dataset_paths.append(glob.glob(data_path + '/*.png'))
    
dataset_paths = list(flatten(dataset_paths))

print('train_image_path example: ', dataset_paths[0])
print('class example: ', classes[0])

print(" Length of Dataset and label ",len(dataset_paths),len(classes))


#######################################################
#      Create dictionary for class indexes
#######################################################

idx_to_class = {i:j for i, j in enumerate(classes)}
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
   

top_paths,test_paths = dataset_paths[:int(0.8*len(dataset_paths))],dataset_paths[int(0.8*len(dataset_paths)):]
train_paths,val_paths = top_paths[:int(0.8*len(top_paths))],top_paths[int(0.8*len(top_paths)):]

#######################################################
#                  Create Dataset
#######################################################
train_dataset = CustomDataset(train_paths,data_transforms)
valid_dataset = CustomDataset(val_paths,data_transforms) #test transforms are applied
test_dataset = CustomDataset(test_paths,data_transforms)

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