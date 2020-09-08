from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from preprocessors import CategoricalEncoding


moa_pipe = Pipeline(
    [
        ('categorical_encoder', CategoricalEncoding)
        ('logistic_regression', LogisticRegression)
    ]
)