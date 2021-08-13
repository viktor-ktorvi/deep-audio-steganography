import os
import json
from constants.constants import CHANNELS

MESSAGE_LEN = 64
BOTTLENECK_CHANNEL_SIZE = CHANNELS['large']
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.00005


def save_parameters(filepath):
    params = {
        'MESSAGE_LEN': MESSAGE_LEN,
        'BOTTLENECK_CHANNEL_SIZE': BOTTLENECK_CHANNEL_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'LEARNING_RATE': LEARNING_RATE
    }

    with open(os.path.join(filepath, 'training parameters.json'), 'w') as fp:
        json.dump(params, fp)
