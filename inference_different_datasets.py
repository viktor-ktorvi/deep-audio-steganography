import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from network_modules.autoencoder import AutoEncoder
from utils.data_loading import get_inference_data
from utils.accuracy import pass_data_through, calc_mean_accuracy

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_FOLDER
from constants.parameters import TRAINING_PARAMETERS_JSON
from constants.constants import DEVICE

MODEL_TO_LOAD = '512 x 4.0 bit'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'
DATASETS = ['birds', 'piano', 'drums', 'speech', 'digits']

NUM_BINS = 30

# SMALL_SIZE = 14
# MEDIUM_SIZE = 15
# BIGGER_SIZE = 16
from inference import SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE

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
        MODEL_FOLDER_PATH = os.path.join(SAVE_MODELS_PATH, MODEL_TO_LOAD)
        MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_NAME + MODEL_EXTENSION)
        PARAMETERS_PATH = os.path.join(MODEL_FOLDER_PATH, MODEL_PARAMETERS_FOLDER)
        INFERENCE_DATA_PATH = os.path.join(INFERENCE_DATA_FOLDER, dataset)

        f = open(os.path.join(PARAMETERS_PATH, TRAINING_PARAMETERS_JSON))
        training_parameters = json.load(f)
        f.close()

        # TODO Really should be saving all the parameters as one json so that I can just pass them in as a dict
        strides = np.load(os.path.join(PARAMETERS_PATH, 'strides.npy'))
        bottleneck_channel_size = training_parameters['BOTTLENECK_CHANNEL_SIZE']
        message_len = training_parameters['MESSAGE_LEN']

        model = AutoEncoder(strides=strides, bottleneck_channel_size=bottleneck_channel_size,
                            message_len=message_len).to(
            DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))

        # %% Loading data
        data = get_inference_data(data_path=INFERENCE_DATA_PATH,
                                  num_signals=None,
                                  high=training_parameters['HIGH'],
                                  bottleneck_channel_size=training_parameters['BOTTLENECK_CHANNEL_SIZE'],
                                  message_len=message_len)

        dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
        with torch.no_grad():
            original_messages, reconstructed_messages, original_audio, modified_audio = pass_data_through(model,
                                                                                                          dataloader,
                                                                                                          DEVICE)
        modified_audio = modified_audio.squeeze()

        # %% Accuracy
        test_acc = calc_mean_accuracy(reconstructed_messages, original_messages, high=training_parameters['HIGH'])

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
        ax[i].hist(x=snrs[i], bins=NUM_BINS, label='histogram')
        ax[i].axvline(x=mean_snrs[i], color='lime', label='mean = {:2.2f} [dB]'.format(mean_snrs[i]))
        ax[i].axvline(x=median_snrs[i], color='orange', label='median = {:2.2f} [dB]'.format(median_snrs[i]))
        ax[i].set_title(DATASETS[i] if DATASETS[i] != 'birds' else 'birds (train set)')
        ax[i].set_xlabel('SNR [dB]')
        ax[i].set_xlim(xmin, xmax)
        ax[i].legend(bbox_to_anchor=(1.3, -0.35))

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
