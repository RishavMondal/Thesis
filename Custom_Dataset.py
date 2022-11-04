import torch
import torchvision.transforms as transforms
import torchvision
 
from torch.utils.data import DataLoader,Dataset
 
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
  def __init__(self,csv,img_folder,transform):
    self.csv=csv
    self.transform=transform
    self.img_folder=img_folder
     
    self.image_names=self.csv[:]['Id']
    self.labels=np.array(self.csv.drop(['Id', 'Genre'], axis=1))
   
#The __len__ function returns the number of samples in our dataset.
  def __len__(self):
    return len(self.image_names)
 
  def __getitem__(self,index):
     
    image=cv2.imread(self.img_folder+self.image_names.iloc[index]+'.jpg')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
 
    image=self.transform(image)
    targets=self.labels[index]
     
    #sample = {'image': image,'labels':targets}
 
    return image,targets