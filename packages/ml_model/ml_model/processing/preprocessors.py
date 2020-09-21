import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ml_model.config import config
from ml_model.nn_ops import train_nn


class BasicPreprocessing(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(X):

        X = pd.get_dummies(X, columns=['cp_dose'], drop_first=True)
        X['cp_time'] = X['cp_time'].map({24: 1, 48: 2, 72: 3})

        return X

