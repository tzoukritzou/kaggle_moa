import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml_model.config import config
import ml_model.nn_ops.nn_config as nn_config
from ml_model.nn_ops.neural_network import Net
from ml_model.nn_ops.nn_data import BasicDataset

from sklearn.metrics import log_loss
import os


def create_tensor(data):

    data = torch.tensor(data).float()
    return data


def create_loaders(X_train, X_test, y_train, y_test):

    X_train, X_test = create_tensor(X_train), create_tensor(X_test)
    y_train, y_test = create_tensor(y_train), create_tensor(y_test)

    train_dataset, test_dataset = BasicDataset(X_train, y_train), BasicDataset(X_test, y_test)
    train_loader, test_loader = DataLoader(train_dataset, batch_size=nn_config.BATCH_SIZE, drop_last=True), \
                                DataLoader(test_dataset, batch_size=nn_config.BATCH_SIZE, drop_last=True)

    return train_loader, test_loader


def save_best_model(metric, best_metric, epoch, model):

    if metric < best_metric:
        best_metric = metric
        torch.save(model.state_dict(), os.path.join(config.TRAINING_RESULTS_DIR, nn_config.NN_NAME))
        print("Saved best model of epoch {}".format(epoch))

    return best_metric


def train_nn(X_train, X_test, y_train, y_test):

    train_loader, test_loader = create_loaders(X_train, X_test, y_train, y_test)

    torch.manual_seed(42)
    model = Net(X_train.shape[1])
    optimizer = nn_config.OPTIMIZER(model.parameters(), lr=nn_config.LR)

    best_kaggle_metric = 100
    for i in range(nn_config.EPOCHS):
        model.train()
        for b, (x_train, y_train) in enumerate(tqdm(train_loader)):

            train_predictions = model(x_train)
            criterion = nn_config.LOSS_FUNCTION
            train_loss = criterion(train_predictions, y_train)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_preds = F.softmax(train_predictions, dim=1).detach().numpy()

        print("Training: Loss for epoch {} is: {} and kaggle metric is: {}".format(i, train_loss,
                                                                                   log_loss(y_train, train_preds)))
        model.eval()
        for b, (x_test, y_test) in enumerate(test_loader):

            test_predictions = model(x_test)
            test_loss = criterion(test_predictions, y_test)
            test_preds = F.softmax(test_predictions, dim=1).detach().numpy()

        kaggle_metric = log_loss(y_test, test_preds)
        best_kaggle_metric = save_best_model(kaggle_metric, best_kaggle_metric, i, model)

        print("Test: Loss for epoch {} is: {} and kaggle metric is: {}".format(i, test_loss,
                                                                               kaggle_metric))
