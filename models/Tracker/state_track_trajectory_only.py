import torch
import torch.nn as nn
from collections import OrderedDict

class StateTrackerTrajectory(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(60, 120),
            nn.ReLU(True),
            nn.Linear(120, 240),
            nn.ReLU(True),
            nn.Linear(240, 480),
            nn.ReLU(True),
            nn.Linear(480, 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50, 20),
            nn.ReLU(True),
            nn.Linear(20, 2)
        )
    
    def forward(self, input):
        return self.network(input)