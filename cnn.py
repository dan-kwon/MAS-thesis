import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import nn  # All neural network modules

class CNN(nn.Module):
    """
    in_channels: number of channels from images (e.g. RGB images would have 3 channels)
    num_classes: number of classes to predict
    """
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,                
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16*32*32, 1000)
        self.fc2 = nn.Linear(1000, 120)
        self.fc3 = nn.Linear(120, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

