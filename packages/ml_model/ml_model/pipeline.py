from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from ml_model.processing import preprocessors as pp
from ml_model.config import config


nn_pipe = Pipeline(
    [
        ('categorical_encoder', pp.CategoricalEncoding()),
    ]
)

shallow_pipe = Pipeline(
    [
        ('categorical_encoder', pp.CategoricalEncoding()),
        ('scaler', MinMaxScaler()),
        ('logistic_regression', LogisticRegression(multi_class='multinomial', random_state=config.RANDOM_STATE,
                                                   verbose=2, n_jobs=-1))
    ]
)
