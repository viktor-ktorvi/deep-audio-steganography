import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

from constants.paths import TRAIN_DATA_PATH, DATA_FILENAME
from constants.constants import HOLDOUT_RATIO
from constants.parameters import MESSAGE_LEN, BOTTLENECK_CHANNEL_SIZE, HIGH

DATA_PATH = TRAIN_DATA_PATH
DATASET_NAME = 'birds'


def reshape_messages(messages, bottleneck_channel_size=BOTTLENECK_CHANNEL_SIZE):
    NUM_MESSAGES = messages.shape[0]
    LEN_MESSAGES = messages.shape[1]
    messages_reshaped = messages.reshape(NUM_MESSAGES, 1, LEN_MESSAGES)
    return np.broadcast_to(messages_reshaped, (NUM_MESSAGES, bottleneck_channel_size, LEN_MESSAGES))


# TODO HIGH really should be a local variable

def scale_messages(messages, high=HIGH):
    return messages / high - (high - 1) / 2 / high


def inverse_scale_messages(messages, high=HIGH):
    return (messages + (high - 1) / 2 / high) * high


def get_dataset(high=HIGH, bottleneck_channel_size=BOTTLENECK_CHANNEL_SIZE):
    data = np.load(os.path.join(DATA_PATH, DATASET_NAME, DATA_FILENAME + '.npy'))

    NUM_SIGNALS = data.shape[0]

    TRAIN_NUM = round(HOLDOUT_RATIO * NUM_SIGNALS)
    TEST_NUM = NUM_SIGNALS - TRAIN_NUM
    VAL_NUM = round(TEST_NUM / 2)
    TEST_NUM -= VAL_NUM

    messages = np.random.randint(low=0, high=high, size=(NUM_SIGNALS, MESSAGE_LEN))
    messages = scale_messages(messages, high=high)
    messages_reshaped = reshape_messages(messages, bottleneck_channel_size=bottleneck_channel_size)

    tensor_dataset = TensorDataset(torch.tensor(data), torch.tensor(messages), torch.tensor(messages_reshaped))

    train_set, validation_and_testing = torch.utils.data.random_split(tensor_dataset, [TRAIN_NUM, TEST_NUM + VAL_NUM])
    test_set, validation_set = torch.utils.data.random_split(validation_and_testing, [TEST_NUM, VAL_NUM])

    return train_set, validation_set, test_set


def get_inference_data(data_path, num_signals='all', high=HIGH,
                       bottleneck_channel_size=BOTTLENECK_CHANNEL_SIZE, message_len=MESSAGE_LEN):
    data = np.load(os.path.join(data_path, DATA_FILENAME + '.npy'))

    if num_signals == 'all':
        NUM_SIGNALS = data.shape[0]
    else:
        NUM_SIGNALS = num_signals

    messages = np.random.randint(low=0, high=high, size=(NUM_SIGNALS, message_len))
    messages = scale_messages(messages, high=high)
    messages_reshaped = reshape_messages(messages, bottleneck_channel_size=bottleneck_channel_size)

    tensor_dataset = TensorDataset(torch.tensor(data), torch.tensor(messages), torch.tensor(messages_reshaped))

    return tensor_dataset


def generate_binary_messages(num_bits, num_messages):
    return np.random.randint(low=0, high=2, size=(num_messages, num_bits))


def binary2decimal(binary_array, packet_len):
    # careful of the //
    decimal_array = np.zeros((binary_array.shape[0], binary_array.shape[1] // packet_len), dtype=np.uint32)

    # terrible triple, effectivly double loop
    for i in range(decimal_array.shape[0]):
        for j in range(decimal_array.shape[1]):
            packet = binary_array[i, j * packet_len:(j + 1) * packet_len]
            decimal_array[i, j] = int("".join(str(x) for x in packet), 2)

    return decimal_array


def grayCode(n):
    # Right Shift the number
    # by 1 taking xor with
    # original number
    return n ^ (n >> 1)


if __name__ == '__main__':
    num_packets = 120
    packet_len = 4
    num_messages = 27
    binary_messages = generate_binary_messages(num_bits=num_packets * packet_len, num_messages=num_messages)
    decimal_messages = binary2decimal(binary_messages, packet_len)

    gray_decimal_messages = grayCode(decimal_messages)

    for i in range(2 ** packet_len):
        print(i, grayCode(i))
