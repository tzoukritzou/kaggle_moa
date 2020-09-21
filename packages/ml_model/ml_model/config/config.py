import os
import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')

TRAINING_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'training_results')

TRAINING_DATA_FILE = 'train_features.csv'
TRAINING_TARGETS = 'train_targets_scored.csv'
NUM_TARGETS = 206

TESTING_DATA_FILE = 'test_features.csv'

TEST_SIZE = 0.2

RANDOM_STATE = 42