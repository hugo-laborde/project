from tqdm import tqdm
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import GaussianBlur
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from model_unet import unet
from prod_dataloader import train_dataloader,test_dataloader
def train(nom):
    loss=nn.MSELoss()
    startLR=10**-3
    endLR=10**-12
    nb_epoch=10000
    b=[24]
    nb_of_batch=len(b)
    loss_history_train=[[],[],[]]
    loss_history_test=[[],[],[]]
    minloss=1
    ngpu=1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = unet().to(device)
    opt=optim.Adam(model.parameters(),lr=startLR)
    sch = lr_scheduler.LinearLR(opt, start_factor=1, end_factor=endLR/startLR, total_iters=nb_epoch)
    stop=0
    for epoch in tqdm(range(1,nb_epoch+1)):
        test_loss=0
        model.train()
        for (i, j) in train_dataloader:
            opt.zero_grad()
            loss_value=loss(model(i.to(device)).to(device),j.to(device))
            loss_value.backward()
            opt.step()
        model.eval()
        for X, y in test_dataloader:
                pred = model(X.to(device)).to(device)
                test_loss += loss(pred, y.to(device)).item()
        test_loss /= len(test_dataloader)
        sch.step()
        if test_loss<minloss:
            minloss=test_loss
            torch.save(model,nom+".pt")
            stop=0
        if epoch%(nb_epoch/10)==0:
             model=torch.load(nom+".pt")
        stop+=1
        if stop > nb_epoch/5:
            break
        loss_history_train[0]+=[loss_value.item()]
        loss_history_test[0]+=[test_loss]
    print("entrainement fini")