import os
import numpy as np
import torch

from paths import TRAIN_DATA_PATH, DATA_FILENAME
from constants import DEVICE

DATA_PATH = TRAIN_DATA_PATH
DATASET = 'birds'

if __name__ == '__main__':
    data = np.load(os.path.join(DATA_PATH, DATASET, DATA_FILENAME + '.npy'))
