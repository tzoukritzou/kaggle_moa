import pandas as pd
from ml_model.config import config
import os
import joblib

from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def load_dataset(file_name):

    df = pd.read_csv(os.path.join(config.DATASET_DIR, file_name))
    return df


def save_dataset(file, file_path):

    file.to_csv(file_path)


def save_file(file, file_path):

    joblib.dump(file, file_path)


def load_file(file_path):

    file = joblib.load(file_path)
    return file


def simple_train_test_split(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42
    )

    return X_train, X_test, y_train, y_test


def stratified_train_test_split(X, y):

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=42)

    for train_index, test_index in msss.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

    return X_train, X_test, y_train, y_test


