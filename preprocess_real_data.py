import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io.wavfile import write

from constants.constants import SIGNAL_LEN, FS
from constants.paths import DATA_FILENAME
from utils.data_loading import get_segments_from_sound

RANDOM_SUBSET_NUM = 50

if __name__ == '__main__':
    libri_data_path = os.path.join(os.getcwd(), 'inference data', 'LibriSpeech', 'test-clean')
    path = os.path.join(libri_data_path, '*', '*', '*.flac')
    filenames = glob.glob(path)

    segments = None
    for filename in tqdm(filenames):
        few_segments = get_segments_from_sound(path=filename, segment_len=SIGNAL_LEN)
        few_segments = few_segments.astype(np.float16)
        if segments is None:
            segments = few_segments
        else:
            segments = np.vstack((segments, few_segments))

    REAL_DATA_PATH = os.path.join(os.getcwd(), 'inference data', 'real data')
    Path(REAL_DATA_PATH).mkdir(parents=True, exist_ok=True)

    random_subset_indices = np.random.randint(low=0, high=segments.shape[0], size=RANDOM_SUBSET_NUM)

    with open(os.path.join(REAL_DATA_PATH, DATA_FILENAME + '.npy'), 'wb') as f:
        np.save(f, segments)

    # %%
    for idx in random_subset_indices:
        write(os.path.join(REAL_DATA_PATH, 'sample' + str(idx) + '.wav'), FS, segments[idx, :])
