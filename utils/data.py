import cv2 as cv
import numpy as np
import torch
import os

from glob import glob
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, folder="/Users/andidemon/Documents/ITTP/Kelas/2023-Genap/Pengenalan Pola/Program/cell/", ):
        self.dataset = []

        # good: 0 bad:1
        for g in glob(folder + "good/*"):
            print(g)
            image = cv.imread(g)
            image = cv.resize(image, (100, 100))

            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.dataset.append([[image/255], 0])

        for b in glob(folder + "bad/*"):
            image = cv.imread(b)
            image = cv.resize(image, (100, 100))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.dataset.append([[image/255], 1])

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

if __name__=="__main__":
    data = Data()