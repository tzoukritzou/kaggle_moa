from sklearn.pipeline import Pipeline

from ml_model.processing import preprocessors as pp


moa_pipe = Pipeline(
    [
        ('categorical_encoder', pp.CategoricalEncoding)
    ]
)