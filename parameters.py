from constants import CHANNELS, SIGNAL_LEN

MESSAGE_LEN = 64
BOTTLENECK_CHANNEL_SIZE = CHANNELS['large']
BATCH_SIZE = 64
STRIDES = [4, 8, 8]

import numpy as np

assert np.prod(STRIDES) == SIGNAL_LEN / MESSAGE_LEN
