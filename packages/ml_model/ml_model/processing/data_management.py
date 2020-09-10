import pandas as pd
from ml_model.config import config
import os
import joblib


def load_dataset(file_name):

    df = pd.read_csv(os.path.join(config.DATASET_DIR, file_name))
    return df


def save_dataset(file, file_path):

    file.to_csv(file_path)


def save_file(file, file_path):

    joblib.dump(file, file_path)


def load_file():

    file = joblib.load(file_path)
    return file


