import numpy as np
import torch
from torch.utils.data import TensorDataset


def reshape_messages(messages, bottleneck_channel_size):
    NUM_MESSAGES = messages.shape[0]
    LEN_MESSAGES = messages.shape[1]
    messages_reshaped = messages.reshape(NUM_MESSAGES, 1, LEN_MESSAGES)
    return np.broadcast_to(messages_reshaped, (NUM_MESSAGES, bottleneck_channel_size, LEN_MESSAGES))


def get_dataset(data_file_path, num_packets, packet_len, bottleneck_channel_size):
    data = np.load(data_file_path)
    NUM_SIGNALS = data.shape[0]

    binary_messages = generate_binary_messages(num_bits=num_packets * packet_len, num_messages=NUM_SIGNALS)
    preprocessed_messages = preprocess_messages(binary_messages, packet_len)
    messages_reshaped = reshape_messages(preprocessed_messages, bottleneck_channel_size=bottleneck_channel_size)

    return TensorDataset(torch.tensor(data), torch.tensor(preprocessed_messages), torch.tensor(messages_reshaped))


def split_dataset(tensor_dataset, holdout_ratio):
    NUM_SIGNALS = len(tensor_dataset)
    TRAIN_NUM = round(holdout_ratio * NUM_SIGNALS)
    TEST_NUM = NUM_SIGNALS - TRAIN_NUM
    VAL_NUM = round(TEST_NUM / 2)
    TEST_NUM -= VAL_NUM

    train_set, validation_and_testing = torch.utils.data.random_split(tensor_dataset, [TRAIN_NUM, TEST_NUM + VAL_NUM])
    test_set, validation_set = torch.utils.data.random_split(validation_and_testing, [TEST_NUM, VAL_NUM])

    return train_set, validation_set, test_set


# TODO napraviti tako da za high=2 bude -0.5, 0.5 i da uvek bude izmedju -0.5 i 0.5
#  ili -1 i 1
def scale_messages(messages, high):
    return messages / high - (high - 1) / 2 / high


def inverse_scale_messages(messages, high):
    return (messages + (high - 1) / 2 / high) * high


def preprocess_messages(binary_messages, packet_len):
    decimal_messages = binary2decimal(binary_messages, packet_len)
    gray_decimal_messages = gray_code(decimal_messages)
    scaled_gray_decimal_messages = scale_messages(messages=gray_decimal_messages, high=2 ** packet_len)

    return scaled_gray_decimal_messages


def postprocess_messages(reconstructed_messages, packet_len):
    reconstructed_gray_messages = inverse_scale_messages(reconstructed_messages, high=2 ** packet_len)
    reconstructed_gray_messages = np.clip(reconstructed_gray_messages, 0, 2 ** packet_len - 1)
    reconstructed_gray_messages = np.round(reconstructed_gray_messages).astype(np.int32)

    reconstructed_decimal_messages = docode_gray_code(reconstructed_gray_messages, packet_len).astype(np.int32)
    reconstructed_binary_messages = decimal2binary(reconstructed_decimal_messages, packet_len)

    return reconstructed_binary_messages


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


def decimal2binary(decimal_array, packet_len):
    binary_array = np.zeros((decimal_array.shape[0], decimal_array.shape[1] * packet_len), dtype=np.uint8)

    for i in range(decimal_array.shape[0]):
        for j in range(decimal_array.shape[1]):
            packet = [int(x) for x in list('{0:0b}'.format(decimal_array[i, j]))]

            # padd with zeros
            while len(packet) < packet_len:
                packet.insert(0, 0)

            binary_array[i, j * packet_len: (j + 1) * packet_len] = packet

    return binary_array


def gray_code(n):
    # Right Shift the number
    # by 1 taking xor with
    # original number
    return n ^ (n >> 1)


def docode_gray_code(gray_array, n_digits):
    code_dict = {}
    for i in range(2 ** n_digits):
        code_dict[gray_code(i)] = i

    decoded_array = np.zeros_like(gray_array)

    for i in range(gray_array.shape[0]):
        for j in range(gray_array.shape[1]):
            decoded_array[i, j] = code_dict[gray_array[i, j]]

    return decoded_array


if __name__ == '__main__':
    num_packets = 120
    packet_len = 5
    num_messages = 27

    binary_messages = generate_binary_messages(num_bits=num_packets * packet_len, num_messages=num_messages)
    preprocessed_messages = preprocess_messages(binary_messages, packet_len)

    reconstructed_binary_messages = postprocess_messages(preprocessed_messages, packet_len)

    assert binary_messages.all() == reconstructed_binary_messages.all()

    print(np.unique(preprocessed_messages))
