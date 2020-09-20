from ml_model.config import config
from ml_model.processing import data_management as dm
from ml_model import pipeline
from ml_model.nn_ops.train_nn import train_nn


def create_datasets(features, targets):
    feature_names = features.columns
    target_names = targets.columns
    merged = features.merge(targets, on='sig_id')
    features = merged[feature_names].drop('sig_id', axis=1)
    targets = merged[target_names].drop('sig_id', axis=1)
    return features, targets


def run_training():

    features = dm.load_dataset(config.TRAINING_DATA_FILE)
    targets = dm.load_dataset(config.TRAINING_TARGETS)
    features, targets = create_datasets(features, targets)

    # X_train, X_test, y_train, y_test = dm.simple_train_test_split(X, targets)
    X_train, X_test, y_train, y_test = dm.stratified_train_test_split(features, targets)

    # train neural network
    #train_nn(X_train, X_test, y_train, y_test)

    # train shallow learning
    pipeline.shallow_pipe.fit(X_train, y_train)


run_training()