import os
import json
import numpy as np
import torch

from torch.utils.data import DataLoader

from network_modules.autoencoder import AutoEncoder
from data_loading import get_inference_data
from utils.accuracy import pass_data_through, calc_mean_accuracy

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_FOLDER
from constants.parameters import TRAINING_PARAMETERS_JSON
from constants.constants import DEVICE

MODEL_TO_LOAD = '64 x 1.0 bit'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'
DATASET = 'birds'

if __name__ == '__main__':
    # %% Loading parameters and the model
    MODEL_FOLDER_PATH = os.path.join(SAVE_MODELS_PATH, MODEL_TO_LOAD)
    MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_NAME + MODEL_EXTENSION)
    PARAMETERS_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_PARAMETERS_FOLDER)
    INFERENCE_DATA_PATH = os.path.join(INFERENCE_DATA_FOLDER, DATASET)

    f = open(os.path.join(PARAMETERS_PATH, TRAINING_PARAMETERS_JSON))
    training_parameters = json.load(f)
    f.close()

    # TODO Really should be saving all the parameters as one json so that I can just pass them in as a dict
    strides = np.load(os.path.join(PARAMETERS_PATH, 'strides.npy'))
    bottleneck_channel_size = training_parameters['BOTTLENECK_CHANNEL_SIZE']

    model = AutoEncoder(strides=strides, bottleneck_channel_size=bottleneck_channel_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # %% Loading data
    data = get_inference_data(data_path=INFERENCE_DATA_PATH,
                              num_signals=None,
                              high=training_parameters['HIGH'],
                              bottleneck_channel_size=training_parameters['BOTTLENECK_CHANNEL_SIZE'])

    dataloader = DataLoader(data, batch_size=len(data), shuffle=True)
    original_messages, reconstructed_messages, original_audio, modified_audio = pass_data_through(model, dataloader)
    modified_audio = modified_audio.squeeze()

    # %% Accuracy
    test_acc = calc_mean_accuracy(reconstructed_messages, original_messages, high=training_parameters['HIGH'])

    print("Test accuracy is {:3.2f} %".format(test_acc * 100))
    # TODO  save the stego next to the original and name it after the model,
    #  see if everything is ok
    #  do some tests as in SNR, spectrograms, noise and quantization seristance
