print("bonjour")
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
import argparse
def list_of_strings(arg):
    return arg[1:-1].split(',')
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("radical",type=str) # le radical du fichier .champs
    parser.add_argument("pfs",type=str) # le prefixe de sortie
    parser.add_argument("lp",type=list_of_strings) # liste des nom des parametre
    parser.add_argument("li",type=list_of_strings) # liste des interval ou faire varier les parametre
    parser.add_argument("nom_model",type=str)
    args = parser.parse_args()
    radical=args.radical
    pfs=args.pfs
    lp=args.lp
    li=args.li
    nom=args.nom_model
    from prod_dataset import Prod_dataset
    Prod_dataset(radical,pfs,lp,li)
    from train import train
    train(nom)
    from test import test
    test(radical,pfs,lp,li,nom)
if __name__ == "__main__":
    main()
