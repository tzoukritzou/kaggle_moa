from ml_model.config import config
from ml_model.processing import data_management as dm
from ml_model import pipeline
from ml_model.nn_ops.train_nn import train_nn
from ml_model.processing.data_management import save_file, save_dataset

import os
import pandas as pd


def run_training(features, targets):

    # features = dm.load_dataset(config.TRAINING_DATA_FILE)
    # targets = dm.load_dataset(config.TRAINING_TARGETS)

    X = pipeline.moa_pipe.transform(features)
    X = X.drop('sig_id', axis=1)
    targets = targets.drop('sig_id', axis=1)

    # train neural network
    preds = train_nn(X.values, targets.values)

    save_dataset(pd.DataFrame(preds), os.path.join(config.TRAINING_RESULTS_DIR, 'preds.csv'))

    return preds

features = dm.load_dataset(config.TRAINING_DATA_FILE)
targets = dm.load_dataset(config.TRAINING_TARGETS)
preds = run_training(features, targets)