import numpy as np
from utils.data_loading import postprocess_messages
from utils.train_utils import pass_data_through


def calc_accuracy(model, dataloader, packet_len, device):
    original_messages_cpu, reconstructed_message_cpu, _, _ = pass_data_through(model, dataloader, device)

    reconstructed_binary_messages = postprocess_messages(reconstructed_message_cpu, packet_len)

    original_binary_messages = postprocess_messages(original_messages_cpu, packet_len)

    predictions = reconstructed_binary_messages == original_binary_messages

    return np.mean(predictions)
