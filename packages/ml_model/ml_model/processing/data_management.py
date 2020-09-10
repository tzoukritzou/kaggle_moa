import pandas as pd
from ml_model.config import config
import os
import joblib


def load_dataset(file_name):

    df = pd.read_csv(os.path.join(config.DATASET_DIR, file_name))
    return df


def save_file(file, file_path):

    joblib.dump(file, file_path)


