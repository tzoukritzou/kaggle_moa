import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_model.config import config


class Net(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(self.in_features, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, config.NUM_TARGETS)

    def forward(self, X):
        x = F.relu(self.fc1(X))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
