import json
import os
import torch


def save_parameters(filepath, params):
    with open(os.path.join(filepath), 'w') as fp:
        json.dump(params, fp)


def pass_data_through(model, dataloader, device):
    for i, data in enumerate(dataloader, 0):
        original_audio, original_messages, original_messages_reshaped = data[0].to(device, dtype=torch.float), data[
            1].to(
            device, dtype=torch.float), data[2].to(device, dtype=torch.float)

        reconstructed_message, modified_audio = model(original_audio.unsqueeze(1), original_messages_reshaped)

        original_messages_cpu = original_messages.detach().cpu().numpy()
        reconstructed_message_cpu = reconstructed_message.detach().cpu().numpy()
        original_audio_cpu = original_audio.detach().cpu().numpy()
        modified_audio_cpu = modified_audio.detach().cpu().numpy()

        return original_messages_cpu, reconstructed_message_cpu, original_audio_cpu, modified_audio_cpu
