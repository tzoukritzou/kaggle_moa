import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ml_model.config import config
from ml_model.nn_ops import train_nn


class CategoricalEncoding(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(X):
        X = pd.get_dummies(X, columns=config.CATEGORICAL_VARS, drop_first=True)
        return X

