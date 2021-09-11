import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from utils.data_loading import get_dataset
from utils.inference_utils import load_saved_model, quantize_array
from utils.train_utils import pass_data_through
from utils.accuracy import calc_mean_accuracy

from constants.paths import SAVE_MODELS_PATH, MODEL_PARAMETERS_FOLDER, INFERENCE_DATA_FOLDER, DATA_FILENAME
from constants.constants import DEVICE, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE

from train import TRAINING_PARAMETERS_JSON

MODEL_TO_LOAD = '512 x 4 bit merged data'
MODEL_NAME = 'autoencoder'
MODEL_EXTENSION = '.pt'
DATASET = 'real data'

BATCH_SIZE = 1000

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
        original_messages, _, _, modified_audio = pass_data_through(model, dataloader,
                                                                    DEVICE)
    modified_audio = modified_audio.squeeze()

    # %% Noise
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
                                      packet_len=training_parameters['PACKET_LEN']) * 100
        test_accs_noise.append(test_acc)
        print("\nTest accuracy {:3.2f} % for relative sigma {:3.1f} %".format(test_acc, percentage * 100))

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
                                      packet_len=training_parameters['PACKET_LEN']) * 100
        test_accs_quant.append(test_acc)
        print("\nTest accuracy {:3.2f} % for {:d} quantization levels".format(test_acc, quant_num))



    # TODO U radu pricati o dva slucaja, kada je trnirano na 1 skupu, radi dobro al slabo generalizuje, a kad je na svim
    #  skupovima onda za veci kapacitet dobro generalizuje ali los kvalitet ima

    # TODO Plotovati i sum i kvantizaciju kao subplot, i onda tako za svaki model, 1x2 i tako 4 puta
    # %%
    height = 1
    width = 2
    marker = 's'
    markersize = 10
    fig, ax = plt.subplots(height, width, sharey='all', sharex='none', tight_layout=True)
    ax[0].grid(b=True, axis='both', linestyle=':')
    ax[0].plot(percentages, test_accs_noise, marker=marker, linestyle=':', markersize=markersize)
    ax[0].set_title('Tačnost rekonstrukcije u prisustvu šuma')
    ax[0].set_xlabel('$\sigma_n/\sigma_0$')
    ax[0].set_ylabel('Tačnost u %')

    ax[1].grid(b=True, axis='both', linestyle=':')
    ax[1].plot(quant_nums, test_accs_quant, marker=marker, linestyle=':', markersize=markersize)
    ax[1].set_title('Tačnost rekonstrukcije pri kvantizaciji')
    ax[1].set_xlabel('br. kvantizacionih nivoa')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
