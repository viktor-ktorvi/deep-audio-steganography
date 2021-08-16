import os
import json

from constants.constants import CHANNELS

MESSAGE_LEN = 64
BOTTLENECK_CHANNEL_SIZE = 25
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.00005
HIGH = 2

TRAINING_PARAMETERS_JSON = 'training parameters.json'


def save_parameters(filepath):
    params = {
        'MESSAGE_LEN': MESSAGE_LEN,
        'BOTTLENECK_CHANNEL_SIZE': BOTTLENECK_CHANNEL_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'HIGH': HIGH
    }

    with open(os.path.join(filepath, TRAINING_PARAMETERS_JSON), 'w') as fp:
        json.dump(params, fp)
