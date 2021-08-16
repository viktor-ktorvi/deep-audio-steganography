import os
import json
import numpy as np
import torch

from network_modules.autoencoder import AutoEncoder

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_PATH
from constants.parameters import TRAINING_PARAMETERS_JSON
from constants.constants import DEVICE

MODEL_TO_LOAD = '64 x 1.0 bit'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'

if __name__ == '__main__':
    MODEL_FOLDER_PATH = os.path.join(SAVE_MODELS_PATH, MODEL_TO_LOAD)
    MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_NAME + MODEL_EXTENSION)
    PARAMETERS_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_PARAMETERS_FOLDER)

    f = open(os.path.join(PARAMETERS_PATH, TRAINING_PARAMETERS_JSON))
    training_parameters = json.load(f)
    f.close()

    # TODO Really should be saving all the parameters as one json so that I can just pass them in as a dict
    strides = np.load(os.path.join(PARAMETERS_PATH, 'strides.npy'))
    bottleneck_channel_size = training_parameters['BOTTLENECK_CHANNEL_SIZE']

    model = AutoEncoder(strides=strides, bottleneck_channel_size=bottleneck_channel_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

