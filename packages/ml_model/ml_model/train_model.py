from ml_model.config import config
from ml_model.processing import data_management as dm
from ml_model import pipeline
from ml_model.nn_ops.train_nn import train_nn
from ml_model import predict

import torch


def create_datasets(features, targets):
    feature_names = features.columns
    target_names = targets.columns
    merged = features.merge(targets, on='sig_id')
    merged = merged[merged.cp_type != 'ctl_vehicle']
    merged = merged.drop('cp_type', axis=1).reset_index(drop=True)
    feature_names = list(feature_names)
    feature_names.remove('cp_type')
    features = merged[feature_names].drop('sig_id', axis=1)
    targets = merged[target_names].drop('sig_id', axis=1)
    return features, targets


def create_weights(y_train):

    targets_sum = y_train.sum(axis=0)
    weights = y_train.shape[0]/(y_train.shape[1]*targets_sum).values
    return torch.from_numpy(weights)


def run_training():

    features = dm.load_dataset(config.TRAINING_DATA_FILE)
    targets = dm.load_dataset(config.TRAINING_TARGETS)
    features, targets = create_datasets(features, targets)

    # X_train, X_test, y_train, y_test = dm.simple_train_test_split(X, targets)
    X_train, X_test, y_train, y_test = dm.stratified_train_test_split(features, targets)
    weights = create_weights(y_train)

    # train neural network
    train_nn(X_train, X_test, y_train, y_test, weights)

    # train shallow learning
    # pipeline.shallow_pipe.fit(X_train, y_train)
    # dm.save_file(pipeline, config.TRAINING_RESULTS_DIR + 'random_forest_pipeline.pkl')
    # predict.evaluate_shallow_model(pipeline, X_train, X_test, y_train, y_test)


run_training()