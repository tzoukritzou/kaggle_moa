import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import config


class CategoricalEncoding(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(self, X):
        X = pd.get_dummies(X, columns=config.CATEGORICAL_VARS, drop_first=True)
        return X
