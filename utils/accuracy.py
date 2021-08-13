import torch
import numpy as np
from constants.constants import DEVICE


def calc_autoencoder_accuracy(model, dataloader):
    for i, data in enumerate(dataloader, 0):
        train_audio, train_messages, train_messages_reshaped = data[0].to(DEVICE, dtype=torch.float), data[1].to(
            DEVICE, dtype=torch.float), data[2].to(DEVICE, dtype=torch.float)

        reconstructed_message, _ = model(train_audio.unsqueeze(1), train_messages_reshaped)

        test_labels_cpu = train_messages.detach().cpu().numpy()
        outputs_cpu = reconstructed_message.detach().cpu().numpy()

        break

    return calc_mean_accuracy(outputs_cpu, np.round(test_labels_cpu)), outputs_cpu, test_labels_cpu


def calc_mean_accuracy(outputs_cpu, test_labels_cpu):
    accuracies = []
    for idx in range(outputs_cpu.shape[0]):
        prediction = [test_labels_cpu[idx, i] == round(outputs_cpu[idx, i]) for i in range(outputs_cpu.shape[1])]
        accuracies.append(sum(prediction) / len(prediction))

    return np.mean(accuracies)
