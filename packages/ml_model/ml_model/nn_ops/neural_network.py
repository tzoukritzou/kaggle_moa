import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from ml_model.config import config


class Net(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        neurons_1 = 512
        neurons_2 = 512
        self.in_features = in_features
        self.fc1 = weight_norm(nn.Linear(self.in_features, neurons_1))
        self.bn1 = nn.BatchNorm1d(num_features=neurons_1)
        self.d1 = nn.Dropout(p=0.5)
        self.fc2 = weight_norm(nn.Linear(neurons_1, neurons_2))
        self.bn2 = nn.BatchNorm1d(num_features=neurons_2)
        self.d2 = nn.Dropout(p=0.5)
        self.fc3 = weight_norm(nn.Linear(neurons_2, config.NUM_TARGETS))

    def forward(self, X):
        x = F.relu(self.bn1(self.fc1(X)))
        x = self.d1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.d2(x)
        x = self.fc3(x)
        return x
