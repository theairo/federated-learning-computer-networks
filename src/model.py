import numpy as np

import torch.nn as nn
import torch.nn.functional as F

# Main model architecture
class mnistNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution layers
        # Input (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
        # -> (N, 10, 26, 26) -> pool -> (N, 10, 13, 13)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)
        # -> (N, 20, 11, 11) -> pool -> (N, 20, 5, 5)

        self.fc1 = nn.Linear(in_features = 20 * 5 * 5, out_features = 50)

        self.out = nn.Linear(in_features = 50, out_features = 10)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 20 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = self.out(x)
        
        return x