import os
import numpy as np
import torch
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from utils.data_loading import get_dataset
from utils.inference_utils import signal_to_noise_ratio, load_saved_model, log_intensity, delete_all_files_in_folder
from utils.train_utils import pass_data_through
from utils.accuracy import calc_mean_accuracy

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_FOLDER, INFERENCE_RESULTS_FOLDER, \
    STEGANOGRAPHIC_AUDIO_FOLDER, ORIGINAL_AUDIO_FOLDER, DATA_FILENAME
from constants.constants import DEVICE, FS, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE

from train import TRAINING_PARAMETERS_JSON

MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'

BATCH_SIZE = 1000

models = ['512 x 1 bit birds', '512 x 1 bit merged data', '512 x 4 bit birds', '512 x 4 bit merged data']
datasets = ['real data', 'birds', 'merged data']

if __name__ == '__main__':
    # %% Seeds
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    for model_name in models:
        # %% Loading parameters and the model
        model_paths = {
            'save_models_path': SAVE_MODELS_PATH,
            'model_to_load': model_name,
            'model_name': MODEL_NAME,
            'model_extension': MODEL_EXTENSION,
            'model_parameters_folder': MODEL_PARAMETERS_FOLDER,
            'training_parameters_json': TRAINING_PARAMETERS_JSON
        }

        model, training_parameters = load_saved_model(**model_paths)
        model = model.to(DEVICE)

        print('Model: ', model_name)
        for dataset in datasets:
            # %% Loading data

            inference_data_parameters = {
                'data_file_path': os.path.join(INFERENCE_DATA_FOLDER, dataset, DATA_FILENAME + '.npy'),
                'num_packets': training_parameters['NUM_PACKETS'],
                'packet_len': training_parameters['PACKET_LEN'],
                'bottleneck_channel_size': training_parameters['BOTTLENECK_CHANNEL_SIZE']
            }

            data = get_dataset(**inference_data_parameters)
            dataloader = DataLoader(data, batch_size=len(data) if len(data) < BATCH_SIZE else BATCH_SIZE, shuffle=False)

            with torch.no_grad():
                original_messages, reconstructed_messages, original_audio, modified_audio = pass_data_through(model,
                                                                                                              dataloader,
                                                                                                              DEVICE)
            modified_audio = modified_audio.squeeze()
            # %% Accuracy
            test_acc = calc_mean_accuracy(original_messages, reconstructed_messages,
                                          packet_len=training_parameters['PACKET_LEN'])

            print('\tDataset: {:s} accuracy =  {:3.2f} %'.format(dataset, test_acc * 100))
