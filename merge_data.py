import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from scipy.io.wavfile import write

from constants.paths import MIXED_DATA_FOLDER, DATA_FILENAME, MERGED_DATASET
from constants.constants import FS

DATASETS = ['birds', 'piano', 'drums', 'speech', 'digits']
DATA_FOLDER = MIXED_DATA_FOLDER
FILENAME = DATA_FILENAME
NEW_DATASET_FOLDER = MERGED_DATASET
RANDOM_SUBSET_NUM = 30

if __name__ == '__main__':
    data = np.load(os.path.join(DATA_FOLDER, DATASETS[0], FILENAME + '.npy'))

    print('Merging...')

    # make sure it's not loading the same thing over again
    for i in tqdm(range(1, len(DATASETS))):
        data = np.vstack((data, np.load(os.path.join(DATA_FOLDER, DATASETS[i], FILENAME + '.npy'))))

    print('Saving results...')

    NEW_DATASET_PATH = os.path.join(DATA_FOLDER, NEW_DATASET_FOLDER)
    Path(NEW_DATASET_PATH).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(NEW_DATASET_PATH, FILENAME + '.npy'), data)

    random_subset_indices = np.random.randint(low=0, high=data.shape[0], size=RANDOM_SUBSET_NUM)
    for i in range(len(random_subset_indices)):
        write(os.path.join(NEW_DATASET_PATH, 'example' + str(i) + '.wav'), FS, data[random_subset_indices[i], :])

    print('Done')
