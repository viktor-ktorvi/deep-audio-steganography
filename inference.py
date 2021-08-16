import os
import json

from network_modules.autoencoder import AutoEncoder

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER
from constants.parameters import TRAINING_PARAMETERS_JSON

MODEL_TO_LOAD = '64 x 3.0 bit'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'

if __name__ == '__main__':
    MODEL_FOLDER_PATH = os.path.join(SAVE_MODELS_PATH, MODEL_TO_LOAD)
    MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_NAME + MODEL_EXTENSION)
    PARAMETERS_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_PARAMETERS_FOLDER)

    f = open(os.path.join(PARAMETERS_PATH, TRAINING_PARAMETERS_JSON))
    training_parameters = json.load(f)
    f.close()
