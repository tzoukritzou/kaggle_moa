import pandas as pd
import os
import torch.nn.functional as F

from sklearn.metrics import log_loss, recall_score, accuracy_score, precision_score

from ml_model.config import config
from ml_model.nn_ops import nn_config
from ml_model import pipeline
from ml_model.processing import data_management as dm
from ml_model.nn_ops.train_nn import create_tensor


def make_predictions(model_name):

    print("Loading test file...")
    test_file = dm.load_dataset(config.TESTING_DATA_FILE)
    test_file = pipeline.moa_pipe.transform(test_file)
    test_file = test_file.drop('sig_id', axis=1)

    model = dm.load_file(os.path.join(config.TRAINING_RESULTS_DIR, model_name))

    # if using neural network
    features = create_tensor(test_file.values)

    print("Making predictions...")
    output = model(features)
    output = F.softmax(output, dim=1)

    return output.detach().numpy()


def create_sub_file(preds):

    print("Creating submission file...")
    targets_names = dm.load_dataset(config.TRAINING_TARGETS).drop('sig_id', axis=1).columns
    ids = dm.load_dataset(config.TESTING_DATA_FILE)['sig_id']

    sub = pd.DataFrame(preds, columns=targets_names)
    sub['sig_id'] = ids
    sub = sub.set_index('sig_id')

    return sub


def predict_for_kaggle():

    preds = make_predictions(nn_config.NN_NAME)
    sub_file = create_sub_file(preds)

    print("Saving submission file...")
    dm.save_dataset(sub_file, os.path.join(config.TRAINING_RESULTS_DIR, 'submission.csv'))


def evaluate_shallow_model(pipeline, X_train, X_test, y_train, y_test):

    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    print("Log loss for train set is {} and log loss for test set is {}".format(log_loss(y_train, train_preds),
                                                                                log_loss(y_test, test_preds)))
    print("Train recall: {} and test recall: {}".format(recall_score(y_train, train_preds),
                                                        recall_score(y_test, test_preds)))
    print("Train accuracy: {} and test accuracy: {}".format(accuracy_score(y_train, train_preds),
                                                            accuracy_score(y_test, test_preds)))
    print("Train precision: {} and test precision: {}".format(precision_score(y_train, train_preds),
                                                              precision_score(y_test, test_preds)))


# predict_for_kaggle()