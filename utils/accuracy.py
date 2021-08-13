import torch
import numpy as np
from constants.constants import DEVICE
from data_loading import inverse_scale_messages

def pass_data_through(model, dataloader):
    for i, data in enumerate(dataloader, 0):
        original_audio, original_messages, original_messages_reshaped = data[0].to(DEVICE, dtype=torch.float), data[
            1].to(
            DEVICE, dtype=torch.float), data[2].to(DEVICE, dtype=torch.float)

        reconstructed_message, modified_audio = model(original_audio.unsqueeze(1), original_messages_reshaped)

        original_messages_cpu = original_messages.detach().cpu().numpy()
        reconstructed_message_cpu = reconstructed_message.detach().cpu().numpy()
        original_audio_cpu = original_audio.detach().cpu().numpy()
        modified_audio_cpu = modified_audio.detach().cpu().numpy()

        return original_messages_cpu, reconstructed_message_cpu, original_audio_cpu, modified_audio_cpu


def calc_accuracy(model, dataloader):
    original_messages_cpu, reconstructed_message_cpu, original_audio_cpu, modified_audio_cpu = pass_data_through(model,
                                                                                                                 dataloader)

    return calc_mean_accuracy(reconstructed_message_cpu, original_messages_cpu)


def calc_mean_accuracy(outputs_cpu, test_labels_cpu):

    accuracies = []

    outputs_cpu = inverse_scale_messages(outputs_cpu)

    test_labels_cpu = inverse_scale_messages(test_labels_cpu)

    for idx in range(outputs_cpu.shape[0]):
        prediction = [test_labels_cpu[idx, i] == round(outputs_cpu[idx, i]) for i in range(outputs_cpu.shape[1])]
        accuracies.append(sum(prediction) / len(prediction))

    return np.mean(accuracies)
