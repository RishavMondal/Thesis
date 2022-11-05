
#hyper params
lr = 1e-4
bs = 32
val_split = 0.85
test_split = 0.90
num_epoch = 10
num_classes = 30
dir_image = './Capitan_score/Capitan_score_uniform'

import fnmatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import numpy as np
import torchvision
from  numpy import exp,absolute
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier as mlp
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score,f1_score,precision_score ,recall_score 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_TVT(path):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(path,transform=data_transforms)
    
    top_size = math.floor(len(dataset)*test_split)
    train_size = math.floor(len(dataset)*val_split)
    val_size = top_size - train_size
    test_size = len(dataset) - top_size
    topset,testset = data.random_split(dataset,lengths=[top_size,test_size])
    trainset,valset = data.random_split(topset,lengths=[train_size,val_size])
    
    print(len(trainset),len(valset),len(testset))
    
    return trainset,valset,testset



def train_model(trainset, valset, model, criterion, optimizer, scheduler, num_epochs):
    dataloaders = {
        'train': data.DataLoader(trainset,batch_size=bs,shuffle=True),
        'val' : data.DataLoader(valset,batch_size=bs,shuffle=True)
    }
    dataset_sizes = {'train':len(trainset),'val':len(valset)}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('bruh')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) 
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def test_acc(model, testset):
    running_corrects = 0
    testloader = data.DataLoader(testset,batch_size=bs,shuffle=True)
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
    return (running_corrects/len(testset))



def get_weighted_score_ft(models,dataset):
    num_models = len(models)
    X = np.empty((0,num_models*num_classes))
    Y = np.empty((0),dtype=int)
    dataloader = data.DataLoader(dataset,batch_size=1,shuffle=True)
    for inputs,labels in dataloader:
        inputs,labels = inputs.to(device),labels.to(device)
        predictions = set()
        with torch.set_grad_enabled(False):
            x = models[0](inputs)
            _, preds = torch.max(x, 1)
            predictions.add(preds)
            for i in range(1,num_models):
                x1 = models[i](inputs)
                _, preds = torch.max(x1, 1)
                predictions.add(preds)
                x = torch.cat((x,x1),dim=1)
            if len(predictions) > 1:
                X = np.append(X,x.cpu().numpy()*3,axis=0)
            else:
                X = np.append(X,x.cpu().numpy(),axis=0)
            Y = np.append(Y,labels.cpu().numpy(),axis=0)     
    return X,Y



def get_models():
    mobile_net = torchvision.models.mobilenet_v2(pretrained=True)
    resnet = torchvision.models.resnet50(pretrained=True)
    

    mobile_net.classifier = nn.Linear(1280,num_classes)
    resnet.fc = nn.Linear(2048,num_classes)
    
    
    mobile_net =  mobile_net.to(device)
    resnet = resnet.to(device)
   
    #return [densenet,googlenet,resnet]
    return [resnet,mobile_net]

def prediction():
    criterion = nn.CrossEntropyLoss()
    ensemble_accuracy=[]

    trainset,valset,testset= get_TVT(dir_image)
    models = get_models()
    for model in models:
        optimizer = optim.Adam(model.parameters(),lr=lr)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)
        model = train_model(trainset, valset, model, criterion, optimizer, exp_lr_scheduler,num_epoch)

        print(test_acc(model,testset))
    
        train_X, train_Y = get_weighted_score_ft(model,trainset)
        test_X, test_Y = get_weighted_score_ft(model,testset)
    
        #predicting using SVM
    model_svc = SVC(decision_function_shape = 'ovr', C = 500, kernel = 'rbf')
    model_svc.fit(train_X,train_Y)
    pred = model_svc.predict(test_X)
    acc = accuracy_score(test_Y, pred)
    ensemble_accuracy.append(acc)
    print('Ensemble accuracy:' +str(acc))
    print("Average Ensemble Accuracy:",sum(ensemble_accuracy)/len(ensemble_accuracy))
    
        #predicting using MLP
    model_mlp = mlp(solver = 'lbfgs', alpha = 1e-5 , random_state = 5, max_iter = 5000)
    model_mlp.fit(train_X,train_Y)
    pred = model_mlp.predict(test_X)
    acc = accuracy_score(test_Y, pred)
    print('Ensemble accuracy:' +str(acc))
    print("Average Ensemble Accuracy:",sum(ensemble_accuracy)/len(ensemble_accuracy))
    

prediction()