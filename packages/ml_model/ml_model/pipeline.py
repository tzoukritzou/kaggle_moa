from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ml_model.processing import preprocessors as pp


moa_pipe = Pipeline(
    [
        ('categorical_encoder', pp.CategoricalEncoding)
    ]
)