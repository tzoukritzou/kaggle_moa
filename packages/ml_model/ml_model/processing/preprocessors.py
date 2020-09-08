import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ml_model.config import config


class CategoricalEncoding(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(x):
        x = pd.get_dummies(x, columns=config.CATEGORICAL_VARS, drop_first=True)
        return x
