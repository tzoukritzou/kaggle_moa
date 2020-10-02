from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler

from lightgbm import LGBMClassifier

from ml_model.processing import preprocessors as pp
from ml_model.config import config


nn_pipe = Pipeline(
    [
        ('basic_preprocessing', pp.BasicPreprocessing()),
        ('min_max_scaler', MinMaxScaler())
    ]
)

shallow_pipe = Pipeline(
    [
        ('basic_preprocessing', pp.BasicPreprocessing()),
        ('scaler', MinMaxScaler()),
        #('logistic_regression', OneVsRestClassifier(LogisticRegression(random_state=config.RANDOM_STATE, verbose=2), n_jobs=-1)),
        #('random_forest', RandomForestClassifier(n_jobs=-1, random_state=config.RANDOM_STATE, verbose=2)),
        ('lightGBM', OneVsRestClassifier(LGBMClassifier(random_state=config.RANDOM_STATE, silent=False)))
    ]
)
