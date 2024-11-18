import os
import numpy as np
from tqdm import tqdm
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import re
import random as rd
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
################################################################################ prod dataset ##################################################################################################################
def Prod_dataset(radical,pfs,lp,li):
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
    os.system("mkdir dataset_gros_carre")
    os.system("mkdir dataset_petit_carre")
    os.system("cp "+radical+".champs dataset_gros_carre")
    os.system("cp "+radical+".champs dataset_petit_carre")
    os.system("cp "+radical+"1-collection.champs dataset_gros_carre")
    os.system("cp "+radical+"2-collection.champs dataset_petit_carre")
    os.system("mv dataset_gros_carre/"+radical+"1-collection.champs dataset_gros_carre/"+radical+"-collection.champs")
    os.system("mv dataset_petit_carre/"+radical+"2-collection.champs dataset_petit_carre/"+radical+"-collection.champs")
    nmax=100
    i=0
    n=10
    vpi=0
    pr=100
    prf=2
    Xp=[]
    Xg=[]
    while pr>prf and i<nmax:
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
        f1=open("dataset_gros_carre/"+radical+".champs","w")
        f1.write(text_l) 
        f1.close()
        f2=open("dataset_petit_carre/"+radical+".champs","w")
        f2.write(text_l) 
        f2.close()
        os.system('cd dataset_gros_carre ; /cluster/logiciels/GIREF/GIREF.AppImage MEF++ ' + radical+" >/dev/null")
        os.system('cd dataset_petit_carre ; /cluster/logiciels/GIREF/GIREF.AppImage MEF++ ' + radical+" >/dev/null")
        fullp = re.search(r"\b({})\b".format(pfs),l[pl])
        outpr=l[pl][fullp.span()[0]:-1]
        path1='dataset_gros_carre/'+outpr+'.cont.vtu'
        path2='dataset_petit_carre/'+outpr+'.cont.vtu'
        xg=extract_vtk(path1)
        xp=extract_vtk(path2)
        Xg+=[xg]
        Xp+=[xp]
        os.system('rm dataset_gros_carre/'+outpr+'.cont.vtu')
        os.system('rm dataset_petit_carre/'+outpr+'.cont.vtu')
        i+=1
        if i%n==0:
            A=torch.stack(tuple(Xg))
            U,S,V = torch.pca_lowrank(A.reshape(i,-1), q=None, center=True, niter=2)
            pr=100*(S[0]-vpi)/S[0]
            vpi=S[0]
            print(pr)
    Xg=torch.stack(tuple(Xg))
    Xp=torch.stack(tuple(Xp))
    torch.save(Xg,"tenseur_grand.pt")
    torch.save(Xp,"tenseur_petit.pt")
    print("dataset produit")