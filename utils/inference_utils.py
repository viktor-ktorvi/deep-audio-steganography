import numpy as np
import torch
import json
import os
import glob

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
    num_packets = training_parameters['NUM_PACKETS']

    model = AutoEncoder(strides=strides, bottleneck_channel_size=bottleneck_channel_size, num_packets=num_packets)
    model.load_state_dict(torch.load(MODEL_PATH))

    return model, training_parameters


def log_intensity(Sxx, k):
    return np.log(1 + k * Sxx)


def delete_all_files_in_folder(PATH):
    files = glob.glob(os.path.join(PATH, '*'))
    for f in files:
        os.remove(f)


def quantize_array(a, num):
    array_min = np.amin(a)
    array_max = np.amax(a)
    quants = np.linspace(start=array_min, stop=array_max, num=num)

    my_bins = np.diff(quants) / 2 + quants[:-1]

    # add min and max to interval but widen the interval a little
    my_bins = np.hstack((array_min + np.sign(array_min) * np.abs(array_min) * 0.01,
                         my_bins,
                         array_max + np.sign(array_max) * np.abs(array_max) * 0.01))

    indices = np.digitize(x=a, bins=my_bins)
    return quants[indices - 1]


if __name__ == '__main__':
    a = np.random.randn(2, 16384)
    q = quantize_array(a, num=8)
