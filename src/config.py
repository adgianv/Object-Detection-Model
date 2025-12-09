import os

# Project root directory
# Assumes this file is located in src/ and the project structure is:
# Project/
#   src/
#     config.py
#   data/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dataset paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Class names as defined in the original notebook
CLASSES = ['TANK', 'IFV', 'APC', 'EV', 'AH', 'TH', 'AAP', 'TA', 'AA', 'TART', 'SPART']

# Training Hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
