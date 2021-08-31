import os
import numpy as np
import torch
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from utils.data_loading import get_dataset
from utils.inference_utils import signal_to_noise_ratio, load_saved_model, log_intensity, delete_all_files_in_folder, \
    quantize_array
from utils.train_utils import pass_data_through
from utils.accuracy import calc_mean_accuracy

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_FOLDER, INFERENCE_RESULTS_FOLDER, \
    STEGANOGRAPHIC_AUDIO_FOLDER, ORIGINAL_AUDIO_FOLDER, DATA_FILENAME
from constants.constants import DEVICE, FS

from train import TRAINING_PARAMETERS_JSON

MODEL_TO_LOAD = '512 x 4 bit mixed'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'
DATASET = 'merged data'

RANDOM_RESULTS_FOLDER = 'random examples'
WORST_SNR_FOLDER = 'worst snr examples'
BEST_SNR_EXAMPLES = 'best snr examples'

BATCH_SIZE = 1000

RANDOM_SUBSET_NUM = 20
NUM_BINS = 30

NUM_WORST = 10
NUM_BEST = 10

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18

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
        'data_file_path': os.path.join(INFERENCE_DATA_FOLDER, DATASET, DATA_FILENAME + '.npy'),
        'num_packets': training_parameters['NUM_PACKETS'],
        'packet_len': training_parameters['PACKET_LEN'],
        'bottleneck_channel_size': training_parameters['BOTTLENECK_CHANNEL_SIZE']
    }

    data = get_dataset(**inference_data_parameters)

    dataloader = DataLoader(data, batch_size=len(data) if len(data) < BATCH_SIZE else BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        original_messages, reconstructed_messages, original_audio, modified_audio = pass_data_through(model, dataloader,
                                                                                                      DEVICE)
    modified_audio = modified_audio.squeeze()

    # %% Accuracy
    test_acc = calc_mean_accuracy(original_messages, reconstructed_messages,
                                  packet_len=training_parameters['PACKET_LEN'])

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

    random_subset_indices = np.random.randint(low=0, high=dataloader.batch_size, size=RANDOM_SUBSET_NUM)

    print('Saving results...', end='')
    for idx in random_subset_indices:
        write(os.path.join(ORIGINAL_RANDOM_PATH, DATASET + str(idx) + '.wav'), FS, original_audio[idx, :])
        write(os.path.join(STEG_RANDOM_PATH, DATASET + str(idx) + '.wav'), FS, modified_audio[idx, :])

    print('Done')

    # %% SNR
    snr, mean_snr, median_snr = signal_to_noise_ratio(original_audio, original_audio - modified_audio)

    plt.figure(tight_layout=True)
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

    # TODO do some tests as in SNR, spectrograms, noise and quantization resistance
    #  test idea: shift signal by n samples cyclicly and see if it recnostructs the message correctly
    # TODO Test on real sound data! Something sampled at 16 kHz and chopped up into 16384 sample chunks

    # %% Spectrogram

    k = 1e7

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

    # TODO Ovde se vidi da nekad krije na odredjenim ucestanostima, to je lose i trebalo bi u radu da pricam da
    #  tu adversarial training moze doci u pricu

    # %% Robustness
    # TODO uradi za vise sigma pa plotuj tacnost

    std = np.std(modified_audio)
    percentages = np.linspace(start=0, stop=1, num=11)
    test_accs_noise = []

    for percentage in percentages:
        noisy_modified_audio = modified_audio + percentage * std * np.random.randn(
            *[shape for shape in modified_audio.shape])

        with torch.no_grad():
            noisy_reconstructed_messages = model.decode(
                torch.tensor(noisy_modified_audio, dtype=torch.float32).unsqueeze(1).to(DEVICE))

        noisy_reconstructed_messages = noisy_reconstructed_messages.detach().cpu().numpy()

        test_acc = calc_mean_accuracy(original_messages, noisy_reconstructed_messages,
                                      packet_len=training_parameters['PACKET_LEN'])
        test_accs_noise.append(test_acc)
        print("\nTest accuracy {:3.2f} % for relative sigma {:3.1f} %".format(test_acc * 100, percentage * 100))

    # %% Plot accuracies in reltion to noise

    # TODO maybe a fancy plot option here, fix anotations

    plt.figure(tight_layout=True)
    plt.plot(percentages, test_accs_noise)
    plt.title('Accuracy in the presence of noise')
    plt.xlabel('sigma_n / \sigma_0')
    plt.ylabel('Accuracy [%]')

    # TODO U radu pricati o dva slucaja, kada je trnirano na 1 skupu, radi dobro al slabo generalizuje, a kad je na svim
    #  skupovima onda za veci kapacitet dobro generalizuje ali los kvalitet ima

    # %% Quantization

    quant_nums = 2 ** np.arange(start=10, stop=5, step=-1)
    test_accs_quant = []

    for quant_num in quant_nums:
        quantized_modified_audio = quantize_array(modified_audio, num=quant_num)
        with torch.no_grad():
            quantization_reconstructed_messages = model.decode(
                torch.tensor(quantized_modified_audio, dtype=torch.float32).unsqueeze(1).to(DEVICE))

        quantization_reconstructed_messages = quantization_reconstructed_messages.detach().cpu().numpy()

        test_acc = calc_mean_accuracy(original_messages, quantization_reconstructed_messages,
                                      packet_len=training_parameters['PACKET_LEN'])
        test_accs_quant.append(test_acc)
        print("\nTest accuracy {:3.2f} % for {:d} quantization levels".format(test_acc * 100, quant_num))

    # %% Plot accuracies in reltion to number of quantization levels

    plt.figure(tight_layout=True)
    plt.plot(quant_nums, test_accs_quant)
    plt.title('Accuracy in the presence of quantization')
    plt.xlabel('no. quantization levels')
    plt.ylabel('Accuracy [%]')

    # TODO Plotovati i sum i kvantizaciju kao subplot, i onda tako za svaki model, 1x2 i tako 4 puta