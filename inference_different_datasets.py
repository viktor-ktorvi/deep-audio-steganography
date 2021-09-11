import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from utils.data_loading import get_dataset
from utils.inference_utils import load_saved_model
from utils.train_utils import pass_data_through
from utils.accuracy import calc_mean_accuracy

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_FOLDER, DATA_FILENAME
from constants.constants import DEVICE

from train import TRAINING_PARAMETERS_JSON
from constants.constants import SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE

MODEL_TO_LOAD = '512 x 4 bit merged data'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'

DATASETS = ['birds', 'merged data', 'real data']
NAMES = ['cvrkut ptica', 'sve klase', 'LibriSpeech']

NUM_BINS = 30
BATCH_SIZE = 640

if __name__ == '__main__':
    # %% Plot specs
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # %% Seeds
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # %%
    snrs = []
    mean_snrs = []
    median_snrs = []
    for dataset in DATASETS:
        # %% Loading parameters and the model
        model_paths = {
            'save_models_path': SAVE_MODELS_PATH,
            'model_to_load': MODEL_TO_LOAD,
            'model_name': MODEL_NAME,
            'model_extension': MODEL_EXTENSION,
            'model_parameters_folder': MODEL_PARAMETERS_FOLDER,
            'training_parameters_json': TRAINING_PARAMETERS_JSON
        }

        model, training_parameters = load_saved_model(**model_paths)
        model = model.to(DEVICE)
        # %% Loading data
        inference_data_parameters = {
            'data_file_path': os.path.join(INFERENCE_DATA_FOLDER, dataset, DATA_FILENAME + '.npy'),
            'num_packets': training_parameters['NUM_PACKETS'],
            'packet_len': training_parameters['PACKET_LEN'],
            'bottleneck_channel_size': training_parameters['BOTTLENECK_CHANNEL_SIZE']
        }

        data = get_dataset(**inference_data_parameters)

        dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            original_messages, reconstructed_messages, original_audio, modified_audio = pass_data_through(model,
                                                                                                          dataloader,
                                                                                                          DEVICE)
        modified_audio = modified_audio.squeeze()

        # %% Accuracy
        test_acc = calc_mean_accuracy(original_messages, reconstructed_messages,
                                      packet_len=training_parameters['PACKET_LEN'])
        print("Test accuracy for {:s} is {:3.2f} %".format(dataset, test_acc * 100))

        # %% SNR
        power_original = np.sum(original_audio ** 2, axis=1) / original_audio.shape[1]

        mse = np.sum((original_audio - modified_audio) ** 2, axis=1) / original_audio.shape[1]

        snr = 10 * np.log10(power_original / mse)
        mean_snr = np.mean(snr)
        median_snr = np.median(snr)

        snrs.append(snr)
        mean_snrs.append(mean_snr)
        median_snrs.append(median_snr)

    # %% Plots
    # TODO ne radi bas savrseno
    xmin = np.amin(snrs) - 0.1 * np.abs(np.amin(snrs))
    # xmin = -20
    xmax = np.amax(snrs) + 0.1 * np.abs(np.amax(snrs))
    width = len(snrs)
    height = 1
    fig, ax = plt.subplots(height, width, sharey='all', sharex='none', tight_layout=True)
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for i in range(len(snrs)):
        ax[i].set_box_aspect(1.7)
        ax[i].hist(x=snrs[i], bins=NUM_BINS, label='histogram', density=True)
        ax[i].axvline(x=mean_snrs[i], color='lime', label='sr. vrednost = {:2.2f} [dB]'.format(mean_snrs[i]))
        ax[i].axvline(x=median_snrs[i], color='orange', label='median = {:2.2f} [dB]'.format(median_snrs[i]))
        ax[i].set_title(NAMES[i])
        ax[i].set_xlabel('SNR [dB]')
        ax[i].set_xlim(xmin, xmax)
        # ax[i].legend(bbox_to_anchor=(1.0, -0.15))
        ax[i].legend(loc='upper left')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
