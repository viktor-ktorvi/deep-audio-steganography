import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

from constants.paths import TRAIN_DATA_PATH, DATA_FILENAME
from constants.constants import HOLDOUT_RATIO
from constants.parameters import MESSAGE_LEN, BOTTLENECK_CHANNEL_SIZE

DATA_PATH = TRAIN_DATA_PATH
DATASET_NAME = 'birds'


def prepare_messages(messages):
    NUM_MESSAGES = messages.shape[0]
    LEN_MESSAGES = messages.shape[1]
    messages_reshaped = messages.reshape(NUM_MESSAGES, 1, LEN_MESSAGES)
    return np.broadcast_to(messages_reshaped, (NUM_MESSAGES, BOTTLENECK_CHANNEL_SIZE, LEN_MESSAGES))


def get_dataset(normalize=False):
    data = np.load(os.path.join(DATA_PATH, DATASET_NAME, DATA_FILENAME + '.npy'))

    data_std = None
    data_mean = None
    if normalize:
        data_std = np.std(data)
        data_mean = np.mean(data)

        data = (data - data_mean) / data_std

    NUM_SIGNALS = data.shape[0]

    TRAIN_NUM = round(HOLDOUT_RATIO * NUM_SIGNALS)
    TEST_NUM = NUM_SIGNALS - TRAIN_NUM
    VAL_NUM = round(TEST_NUM / 2)
    TEST_NUM -= VAL_NUM

    messages = np.random.randint(low=0, high=2, size=(NUM_SIGNALS, MESSAGE_LEN))
    messages_reshaped = prepare_messages(messages)

    tensor_dataset = TensorDataset(torch.tensor(data), torch.tensor(messages), torch.tensor(messages_reshaped))

    train_set, validation_and_testing = torch.utils.data.random_split(tensor_dataset, [TRAIN_NUM, TEST_NUM + VAL_NUM])
    test_set, validation_set = torch.utils.data.random_split(validation_and_testing, [TEST_NUM, VAL_NUM])

    return train_set, validation_set, test_set, data_mean, data_std


if __name__ == '__main__':
    train_set, validation_set, test_set, data_mean, data_std = get_dataset()
    print(len(train_set))
