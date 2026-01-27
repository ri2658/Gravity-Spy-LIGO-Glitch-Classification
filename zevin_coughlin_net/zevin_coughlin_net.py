import torch
from torch import nn
import torch.nn.functional as F

class ZevinCoughlinNet(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=5
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=5
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 20 * 25, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x