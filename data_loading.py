import os
import numpy as np

from paths import TRAIN_DATA_PATH, DATA_FILENAME

DATA_PATH = TRAIN_DATA_PATH
DATASET = 'birds'

if __name__ == '__main__':
    data = np.load(os.path.join(DATA_PATH, DATASET, DATA_FILENAME + '.npy'))
