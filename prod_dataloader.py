from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import GaussianBlur
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch
import numpy as np
class myImageDataset(Dataset):
    def __init__(self, Xf,yf ):
        self.X = torch.load(Xf)
        self.y = torch.load(yf)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        y = self.y[idx]
        X = self.X[idx]
        X=torch.nn.functional.interpolate(X.unsqueeze(0), size=y.shape[1:], scale_factor=None, mode='bicubic', align_corners=None, recompute_scale_factor=None, antialias=False)
        return  X[0],y
Data=myImageDataset("tenseur_grand.pt","tenseur_petit.pt")
batch_size = 24
validation_split = .2
shuffle_dataset = True
random_seed= 42
dataset_size = len(Data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, test_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_dataloader = DataLoader(Data, batch_size=batch_size,sampler=train_sampler)
test_dataloader = DataLoader(Data, batch_size=batch_size,sampler=test_sampler)