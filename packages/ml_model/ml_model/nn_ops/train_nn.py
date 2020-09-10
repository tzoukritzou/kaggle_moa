import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import ml_model.nn_ops.nn_config as nn_config
from ml_model.nn_ops.neural_network import Net
from ml_model.nn_ops.nn_data import BasicDataset

from sklearn.metrics import log_loss
import os


def train_nn(X, targets):

    X = torch.tensor(X).float()
    targets = torch.tensor(targets).float()

    dataset = BasicDataset(X, targets)
    loader = DataLoader(dataset, batch_size = nn_config.BATCH_SIZE)
    print(len(dataset))

    torch.manual_seed(42)
    model = Net(X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for i in range(nn_config.EPOCHS):
        for b, (X_train, y_train) in enumerate(tqdm(loader)):

            predictions = model(X_train)
            criterion = nn_config.LOSS_FUNCTION
            loss = criterion(predictions, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = F.softmax(predictions, dim=1).detach().numpy()

        print("Loss for epoch {} is: {} and kaggle metric is: {}".format(i, loss,
                                                                         log_loss(y_train, preds)))

    return preds

