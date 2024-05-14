import numpy as npp
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from utils.data import Data

import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 64)  # Adjust this based on the actual size after conv/pool
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():

    BATCH_SIZE = 4
    EPOCH = 100

    train_loader = DataLoader(Data(folder="/Users/andidemon/Documents/ITTP/Kelas/2023-Genap/Pengenalan Pola/Program/cell/"), batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()  # Use nn.MSELoss() for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCH):
        for batch, (src, trg) in enumerate(train_loader):
            # print(src.shape)
            src = torch.permute(src, (0, 1, 3, 2))
            # print(src.shape)
            pred = model(src)#.to("cpu")

            loss = criterion(pred, trg)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("loss = ", loss)








if __name__=="__main__":
    main()
