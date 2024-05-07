import torch
from torch import nn


class DrivingModel(nn.Module):
    def __init__(self) -> None:
        super(DrivingModel, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 1164),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
        )

    def forward(self, x):
        return self.network(x)