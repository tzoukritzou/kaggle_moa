import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_model.config import config


class Net(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, config.NUM_TARGETS)

    def forward(self, X):
        print(X.shape)
        x = self.fc1(X)
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = F.relu(self.fc3(x))
        return x
