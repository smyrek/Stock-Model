import os

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import pandas as pd
import numpy as np


class StockDataset(Dataset):
    def __init__(self, Datadir):
        self.filelist = [
            Datadir+"/"+f for f in os.listdir(Datadir) if f.endswith("csv")]
        self.x = torch.zeros(200, len(self.filelist), 5)
        self.y = torch.zeros(200, len(self.filelist), 3)

        for i, file in enumerate(self.filelist):
            n = pd.read_csv(file).to_numpy()[:, 1:][::-1]

            np_x = np.array(n, dtype=np.float32)
            np_y = np.array(n[:], dtype=np.float32)[:, 0]

            tensor_x = torch.FloatTensor(np_x)
            tensor_y = torch.FloatTensor(np_y)

            for j in range(200):
                self.x[j][i] = tensor_x[j]
                self.y[j][i] = tensor_y[j+1:j+4] if j < 197 else torch.zeros(3)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)-3


if __name__ == "__main__":
    t = StockDataset("data")

    print("length =", len(t))
    for x, y in t:
        print(x.shape, y.shape)
        break
