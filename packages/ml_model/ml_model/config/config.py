import os
import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')

TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')

TRAINING_DATA_FILE = 'train_features.csv'
TRAINING_TARGETS = 'train_targets_scored.csv'
NUM_TARGETS = 206

TESTING_DATA_FILE = 'test_features.csv'

CATEGORICAL_VARS = ['cp_type', 'cp_dose']
