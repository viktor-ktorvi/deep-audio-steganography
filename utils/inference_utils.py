import numpy as np
import torch
import json
import os

from network_modules.autoencoder import AutoEncoder


def signal_to_noise_ratio(signal_array, noise_array):
    power_original = np.sum(signal_array ** 2, axis=1) / signal_array.shape[1]

    mse = np.sum(noise_array ** 2, axis=1) / signal_array.shape[1]

    snr = 10 * np.log10(power_original / mse)
    mean_snr = np.mean(snr)
    median_snr = np.median(snr)

    return snr, mean_snr, median_snr


def load_saved_model(save_models_path, model_to_load, model_name, model_extension, model_parameters_folder,
                     training_parameters_json):
    MODEL_FOLDER_PATH = os.path.join(save_models_path, model_to_load)
    MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, model_name + model_extension)
    PARAMETERS_PATH = os.path.join(MODEL_FOLDER_PATH, model_parameters_folder)

    f = open(os.path.join(PARAMETERS_PATH, training_parameters_json))
    training_parameters = json.load(f)
    f.close()

    strides = training_parameters['STRIDES']
    bottleneck_channel_size = training_parameters['BOTTLENECK_CHANNEL_SIZE']
    message_len = training_parameters['MESSAGE_LEN']

    model = AutoEncoder(strides=strides, bottleneck_channel_size=bottleneck_channel_size, message_len=message_len)
    model.load_state_dict(torch.load(MODEL_PATH))

    return model, training_parameters


def log_intensity(Sxx, k):
    return np.log(1 + k * Sxx)
