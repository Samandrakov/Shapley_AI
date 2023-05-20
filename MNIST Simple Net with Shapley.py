import torch
import torch.nn as nn
from torch import autograd

class ShapleyNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.weights == nn.Parameter(torch.randn(784))

    def forwad(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = x * self.weights

        x = self.fc3(x)
        return x

model = ShapleyNet()