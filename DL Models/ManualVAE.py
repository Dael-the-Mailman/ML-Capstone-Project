import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from dotenv import dotenv_values
from torch.utils.data import Dataset, DataLoader

config = dotenv_values('../.env')
FEATURES = [str(c) for c in range(1326)]

class ImputedDataset(Dataset):
    def __init__(self, path, chunksize=10, nb_samples=110):
        self.path = path
        self.chunksize = chunksize
        self.len = nb_samples // self.chunksize
    
    def __getitem__(self, idx):
        x = next(
            pd.read_csv(
                self.path,
                skiprows=idx * self.chunksize + 1,
                chunksize=self.chunksize
            )
        )
        x = torch.from_numpy(x.values)
        return x
    
    def __len__(self):
        return self.len

dataset = ImputedDataset(config["ENGINEERED_DATA"] + "imputed_train.csv")
loader = DataLoader(dataset, 10, num_workers=1, shuffle=False)

for batch_idx, data in enumerate(loader):
    print('batch: {}\tdata: {}'.format(batch_idx, data))
