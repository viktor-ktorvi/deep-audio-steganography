import os
import json
import numpy as np
import torch
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from network_modules.autoencoder import AutoEncoder
from data_loading import get_inference_data
from utils.accuracy import pass_data_through, calc_mean_accuracy
from utils.delete_all_files_in_folder import delete_all_files_in_folder

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_FOLDER, INFERENCE_RESULTS_FOLDER, \
    STEGANOGRAPHIC_AUDIO_FOLDER, ORIGINAL_AUDIO_FOLDER
from constants.parameters import TRAINING_PARAMETERS_JSON
from constants.constants import DEVICE, FS

MODEL_TO_LOAD = '512 x 1.0 bit'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'
DATASET = 'birds'

RANDOM_RESULTS_FOLDER = 'random examples'
WORST_SNR_FOLDER = 'worst snr examples'
BEST_SNR_EXAMPLES = 'best snr examples'

RANDOM_SUBSET_NUM = 20
NUM_BINS = 30

NUM_WORST = 10
NUM_BEST = 10

if __name__ == '__main__':
    # %% Seeds
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
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
    message_len = training_parameters['MESSAGE_LEN']

    model = AutoEncoder(strides=strides, bottleneck_channel_size=bottleneck_channel_size, message_len=message_len).to(
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
        original_messages, reconstructed_messages, original_audio, modified_audio = pass_data_through(model, dataloader)
    modified_audio = modified_audio.squeeze()

    # %% Accuracy
    test_acc = calc_mean_accuracy(reconstructed_messages, original_messages, high=training_parameters['HIGH'])

    print("Test accuracy is {:3.2f} %".format(test_acc * 100))

    # %% Saving some examples
    STEG_PATH = os.path.join(INFERENCE_RESULTS_FOLDER, STEGANOGRAPHIC_AUDIO_FOLDER)
    STEG_RANDOM_PATH = os.path.join(STEG_PATH, RANDOM_RESULTS_FOLDER)

    ORIGINAL_PATH = os.path.join(INFERENCE_RESULTS_FOLDER, ORIGINAL_AUDIO_FOLDER)
    ORIGINAL_RANDOM_PATH = os.path.join(ORIGINAL_PATH, RANDOM_RESULTS_FOLDER)

    Path(INFERENCE_RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(STEG_RANDOM_PATH).mkdir(parents=True, exist_ok=True)
    Path(ORIGINAL_RANDOM_PATH).mkdir(parents=True, exist_ok=True)

    delete_all_files_in_folder(STEG_RANDOM_PATH)
    delete_all_files_in_folder(ORIGINAL_RANDOM_PATH)

    random_subset_indices = np.random.randint(low=0, high=len(data), size=RANDOM_SUBSET_NUM)

    print('Saving results...')
    for idx in random_subset_indices:
        write(os.path.join(ORIGINAL_RANDOM_PATH, DATASET + str(idx) + '.wav'), FS, original_audio[idx, :])
        write(os.path.join(STEG_RANDOM_PATH, DATASET + str(idx) + '.wav'), FS, modified_audio[idx, :])

    print('Done')

    # %% SNR
    power_original = np.sum(original_audio ** 2, axis=1) / original_audio.shape[1]

    mse = np.sum((original_audio - modified_audio) ** 2, axis=1) / original_audio.shape[1]

    snr = 10 * np.log10(power_original / mse)
    mean_snr = np.mean(snr)
    median_snr = np.median(snr)

    plt.figure()
    plt.hist(x=snr, bins=NUM_BINS, label='histogram')
    plt.axvline(x=mean_snr, color='lime', label='mean')
    plt.axvline(x=median_snr, color='orange', label='median')
    plt.title('Histogram SNR')
    plt.xlabel('SNR [dB]')
    plt.legend()

    indices_sorted_snr = np.argsort(snr)
    worst_snr_indices = indices_sorted_snr[0:NUM_WORST]
    best_snr_indices = indices_sorted_snr[-NUM_BEST:][::-1]

    ORIGINAL_WORST_EXAMPLES_PATH = os.path.join(ORIGINAL_PATH, WORST_SNR_FOLDER)
    STEG_WORST_EXAMPLES_PATH = os.path.join(STEG_PATH, WORST_SNR_FOLDER)

    ORIGINAL_BEST_EXAMPLES_PATH = os.path.join(ORIGINAL_PATH, BEST_SNR_EXAMPLES)
    STEG_BEST_EXAMPLES_PATH = os.path.join(STEG_PATH, BEST_SNR_EXAMPLES)

    Path(ORIGINAL_WORST_EXAMPLES_PATH).mkdir(parents=True, exist_ok=True)
    Path(STEG_WORST_EXAMPLES_PATH).mkdir(parents=True, exist_ok=True)
    Path(ORIGINAL_BEST_EXAMPLES_PATH).mkdir(parents=True, exist_ok=True)
    Path(STEG_BEST_EXAMPLES_PATH).mkdir(parents=True, exist_ok=True)

    # TODO Posto su losi oni cija je snaga originalno nikakva mozda treba predprocesirati podatke u treningu
    #  da se takvi odstrane, ili mozda da se razvuku tako da svi budu izmedju -1 i 1

    for idx in worst_snr_indices:
        write(os.path.join(ORIGINAL_WORST_EXAMPLES_PATH, DATASET + str(idx) + '.wav'), FS, original_audio[idx, :])
        write(os.path.join(STEG_WORST_EXAMPLES_PATH, DATASET + str(idx) + '.wav'), FS, modified_audio[idx, :])

    for idx in best_snr_indices:
        write(os.path.join(ORIGINAL_BEST_EXAMPLES_PATH, DATASET + str(idx) + '.wav'), FS, original_audio[idx, :])
        write(os.path.join(STEG_BEST_EXAMPLES_PATH, DATASET + str(idx) + '.wav'), FS, modified_audio[idx, :])

    # %% PSNR
    #
    # # TODO Not sure if I'm doing this right
    #
    # min_original = np.amin(original_audio)
    # min_modified = np.amin(modified_audio)
    #
    # max_original = np.amax(original_audio - min_original)
    # max_modified = np.amax(modified_audio - min_modified)
    # max_val = np.max([max_modified, max_original])
    #
    # pmse = np.sum((original_audio - min_original - modified_audio + min_modified) ** 2, axis=1) / original_audio.shape[
    #     1]
    #
    # psnr = 10 * np.log10(max_val ** 2 / pmse)
    # mean_psnr = np.mean(psnr)
    # median_psnr = np.median(psnr)
    #
    # plt.figure()
    # plt.hist(x=psnr, bins=30, label='histogram')
    # plt.axvline(x=mean_psnr, color='lime', label='mean')
    # plt.axvline(x=median_psnr, color='orange', label='median')
    # plt.title('Histogram PSNR')
    # plt.xlabel('PSNR [dB]')
    # plt.legend()

    # TODO SNR and PSNR, to find peak do MAX of original and stego
    # TODO do some tests as in SNR, spectrograms, noise and quantization seristance
    #  histogrm of SNR because some are really good, some are trash (maybe train on more data)
    #  find the trash SNR examples and listen to them
    #  test idea: shift signal by n samples cyclicly and see if it recnostructs the message correctly

    # %% Spectrogram

    k = 1e7
    log_intensity = lambda Sxx, k: np.log(1 + k * Sxx)

    indices = [best_snr_indices[0],
               indices_sorted_snr[round(len(indices_sorted_snr) / 2) + np.random.randint(low=-5, high=5)],
               worst_snr_indices[0]]
    print("\nSNR of selected signals: ", snr[indices], " [dB]")

    height = 3
    width = 3
    fig, ax = plt.subplots(height, width, sharey='all', sharex='all', tight_layout=True)
    ax_titles = ['Original', 'Modified', 'Difference']

    for i in range(3):

        f_axis, t_axis, original_Sxx = spectrogram(original_audio[indices[i], :], FS)
        _, _, modified_Sxx = spectrogram(modified_audio[indices[i], :], FS)
        difference_Sxx = np.abs(original_Sxx - modified_Sxx)

        spectrograms = [original_Sxx, modified_Sxx, difference_Sxx]

        for j in range(len(spectrograms)):
            ax[i, j].pcolormesh(t_axis, f_axis, log_intensity(spectrograms[j], k), shading='gouraud')

            # ax[i, j].set_box_aspect(1)
            if i == 0:
                ax[i, j].set_title(ax_titles[j])

            if i == 2:
                ax[i, j].set_xlabel('t [s]')

            if j == 0:
                ax[i, j].set_ylabel('f [Hz]')

    # https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
