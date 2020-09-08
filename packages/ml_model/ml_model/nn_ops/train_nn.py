import torch

import ml_model.nn_ops.nn_config as nn_config
from ml_model.nn_ops.neural_network import Net

def train_nn(X, targets):

    X = torch.tensor(X)
    targets = torch.tensor(targets)

    model = Net(X.shape[0])

    for i in range(nn_config.EPOCHS):

        predictions = model(X.float())

        criterion = nn_config.LOSS
        loss = criterion(predictions, targets)
        model.zero_grad()
        loss.backward()
        print(loss)