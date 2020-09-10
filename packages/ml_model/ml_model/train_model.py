from ml_model.config import config
from ml_model.processing import data_management as dm
from ml_model import pipeline
from ml_model.nn_ops.train_nn import train_nn


def run_training(features, targets):

    #features = dm.load_dataset(config.TRAINING_DATA_FILE)
    #targets = dm.load_dataset(config.TRAINING_TARGETS)

    X = pipeline.moa_pipe.transform(features)
    X = X.drop('sig_id', axis=1)
    targets = targets.drop('sig_id', axis=1)

    #train neural network
    train_nn(X.values, targets.values)

features = dm.load_dataset(config.TRAINING_DATA_FILE)
targets = dm.load_dataset(config.TRAINING_TARGETS)
run_training(features, targets)