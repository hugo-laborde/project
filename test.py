import os
import numpy as np
from tqdm import tqdm
import time
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
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import re
import random as rd
########################################################################################## model ###########################################################################################################
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
class unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(4, 32)
        self.e2 = encoder_block(32,64)
        self.e3 = encoder_block(64,128)
        self.e4 = encoder_block(128,256)
        self.b = conv_block(256,512)
        self.d1 = decoder_block(512,256)
        self.d2 = decoder_block(256,128)
        self.d3 = decoder_block(128,64)
        self.d4 = decoder_block(64,32)
        self.outputs = nn.Conv2d(32, 4, kernel_size=1, padding=0)
    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs
##################################################################### fonction pour produire le dataset ##############################################################################################
def extract_vtk(file_name):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    # Get the coordinates of nodes in the mesh
    n=reader.GetOutput().GetPointData().GetNumberOfArrays()
    lc=[]
    for i in range(n):
        champs=torch.tensor(vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(i)))
        if len(champs.shape)>1:
            lc+=list(torch.split(champs,1,dim=1))
        else:
            lc+=[champs]
    lc=[(x.reshape(-1)-x.reshape(-1).mean())/x.reshape(-1).std() for x in lc]
    return lc
################################################################################ prod test dataset ##################################################################################################################
def test(radical,pfs,lp,li,nom):
    ll=[]
    f=open(radical+".champs") 
    l=f.readlines()
    pl=0
    for i in range(len(l)):
        for param in lp:
            if " "+param+" " in l[i]:
                ll+=[i]
        if "prefixe_sorties" in l[i]:
            pl=i
    liste_interval = []
    for i in range(len(li)-2):
        if li[i] == "uniform":
            liste_interval+=[["uniform",float(li[i+1]),float(li[i+2])]]
        if li[i] == "normal":
            liste_interval+=[["normal",float(li[i+1]),float(li[i+2])]]
    li=liste_interval
    f.close()
    os.system("mkdir dataset_test_gros_carre")
    os.system("mkdir dataset_test_petit_carre")
    os.system("cp "+radical+".champs dataset_test_gros_carre")
    os.system("cp "+radical+".champs dataset_test_petit_carre")
    os.system("cp "+radical+"1-collection.champs dataset_test_gros_carre")
    os.system("cp "+radical+"2-collection.champs dataset_test_petit_carre")
    os.system("mv dataset_test_gros_carre/"+radical+"1-collection.champs dataset_test_gros_carre/"+radical+"-collection.champs")
    os.system("mv dataset_test_petit_carre/"+radical+"2-collection.champs dataset_test_petit_carre/"+radical+"-collection.champs")
    tens=torch.load("tenseur_grand.pt")
    n=int(tens.shape[0]*0.2)
    Xp=[]
    Xg=[]
    for i in tqdm(range(n)):
        f=open(radical+".champs")
        l=f.readlines()
        l[pl]="prefixe_sorties   "+pfs+"["
        for line in range(len(ll)):
            index = re.search(r"\b({})\b".format(lp[line]),l[ll[line]])
            if li[line][0] == "uniform":
                coef = rd.uniform(li[line][1],li[line][2])
            if li[line][0] == "normal":
                coef = rd.gauss(li[line][1],li[line][2])
            l[ll[line]]=l[ll[line]][:index.span()[0]]+lp[line]+" "+str(coef)+"\n"
            l[pl]+=str(coef)+","
        l[pl]+="]"+"\n"
        text_l="" 
        for x in l: 
            text_l+=x 
        f.close() 
        f1=open("dataset_test_gros_carre/"+radical+".champs","w")
        f1.write(text_l) 
        f1.close()
        f2=open("dataset_test_petit_carre/"+radical+".champs","w")
        f2.write(text_l) 
        f2.close()
        os.system('cd dataset_test_gros_carre ; /cluster/logiciels/GIREF/GIREF.AppImage MEF++ ' + radical+" >/dev/null")
        os.system('cd dataset_test_petit_carre ; /cluster/logiciels/GIREF/GIREF.AppImage MEF++ ' + radical+" >/dev/null")
        fullp = re.search(r"\b({})\b".format(pfs),l[pl])
        outpr=l[pl][fullp.span()[0]:-1]
        path1='dataset_test_gros_carre/'+outpr+'.cont.vtu'
        path2='dataset_test_petit_carre/'+outpr+'.cont.vtu'
        xg=extract_vtk(path1)
        xp=extract_vtk(path2)
        Xg+=[xg]
        Xp+=[xp]
        os.system('rm dataset_test_gros_carre/'+outpr+'.cont.vtu')
        os.system('rm dataset_test_petit_carre/'+outpr+'.cont.vtu')
    Xg=torch.stack(tuple(Xg))
    Xp=torch.stack(tuple(Xp))
    torch.save(Xg,"tenseur_test_grand.pt")
    torch.save(Xp,"tenseur_test_petit.pt")
    print("dataset test produit")
    # test du model avec le dataset de test
    unet_model=torch.load(nom+".pt")
    l=[]
    unet_model.eval()
    norm2=nn.MSELoss()
    for j in range(Xg.shape[0]):
        y=Xp[j]
        yml=unet_model(torch.nn.functional.interpolate(Xg[j].unsqueeze(0), size=None, scale_factor=4, mode='bicubic', align_corners=None, recompute_scale_factor=None, antialias=False))[0]
        l+=[float(norm2(y,yml).detach().numpy())]
    fnorm=open("unet_norm_l2.txt","w")
    fnorm.write(str(l))
    fnorm.close()
    print(np.mean(l))
