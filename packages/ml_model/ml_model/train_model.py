from ml_model.config import config
from ml_model.processing import data_management as dm
from ml_model import pipeline
from ml_model.nn_ops.train_nn import train_nn


def run_training():

    features = dm.load_dataset(config.TRAINING_DATA_FILE)
    targets = dm.load_dataset(config.TRAINING_TARGETS)

    X = pipeline.moa_pipe.transform(features)
    X = X.drop('sig_id', axis=1).values
    targets = targets.drop('sig_id', axis=1).values

    # X_train, X_test, y_train, y_test = dm.simple_train_test_split(X, targets)
    X_train, X_test, y_train, y_test = dm.stratified_train_test_split(X, targets)

    # train neural network
    train_nn(X_train, X_test, y_train, y_test)


run_training()