from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

from ml_model.processing import preprocessors as pp
from ml_model.config import config


nn_pipe = Pipeline(
    [
        ('basic_preprocessing', pp.BasicPreprocessing()),
    ]
)

shallow_pipe = Pipeline(
    [
        ('basic_preprocessing', pp.BasicPreprocessing()),
        #('scaler', MinMaxScaler()),
        #('logistic_regression', LogisticRegression(multi_class='multinomial', random_state=config.RANDOM_STATE,
        #                                           verbose=2, n_jobs=-1)),
        #('random_forest', RandomForestClassifier(n_jobs=-1, random_state=config.RANDOM_STATE, verbose=2)),
        ('lightGBM', LGBMClassifier(objective='multiclass', random_state=config.RANDOM_STATE))
    ]
)
