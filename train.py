import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants.constants import DEVICE, FS
from constants.paths import SAVE_MODELS_PATH, MODEL_NAME, MODEL_EXTENSION, AUDIO_FOLDER, ORIGINAL_AUDIO_FOLDER, \
    STEGANOGRAPHIC_AUDIO_FOLDER, MODEL_PARAMETERS_FOLDER, TRAIN_DATA_PATH, DATA_FILENAME, TRAINING_PARAMETERS_JSON
from loss.autoencoder_loss import AutoEncoderLoss
from network_modules.autoencoder import AutoEncoder
from utils.accuracy import pass_data_through, calc_accuracy
from utils.data_loading import get_dataset, split_dataset
from utils.train_utils import save_parameters

DATASET_NAME = 'birds'
HOLDOUT_RATIO = 0.8

VALIDATION_BATCH_SIZE = 100
WAV_SAVING_NUM = 30

PACKET_LEN = 4
NUM_PACKETS = 512

STRIDES = [4, 4, 2]
BOTTLENECK_CHANNEL_SIZE = 25
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.00005

MODEL_FOLDER_NAME = str(NUM_PACKETS) + ' x ' + str(PACKET_LEN) + ' bit'
MODEL_PATH = os.path.join(SAVE_MODELS_PATH, MODEL_FOLDER_NAME)
ORIGINAL_AUDIO_PATH = os.path.join(MODEL_PATH, AUDIO_FOLDER, ORIGINAL_AUDIO_FOLDER)
STEGANOGRAPHIC_AUDIO_PATH = os.path.join(MODEL_PATH, AUDIO_FOLDER, STEGANOGRAPHIC_AUDIO_FOLDER)
MODEL_PARAMETERS_PATH = os.path.join(MODEL_PATH, MODEL_PARAMETERS_FOLDER)

if __name__ == '__main__':
    # %% Seeds
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # %% Loading the data
    # TODO normalizing seems to make it worse, why? See what people who work with timeseries' do.

    dataset_parameters = {
        'data_file_path': os.path.join(TRAIN_DATA_PATH, DATASET_NAME, DATA_FILENAME + '.npy'),
        'num_packets': NUM_PACKETS,
        'packet_len': PACKET_LEN,
        'bottleneck_channel_size': BOTTLENECK_CHANNEL_SIZE
    }

    tensor_dataset = get_dataset(**dataset_parameters)
    train_set, validation_set, test_set = split_dataset(tensor_dataset, holdout_ratio=HOLDOUT_RATIO)
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    train_std, train_mean = torch.std_mean(train_set.dataset.tensors[0], unbiased=False)
    print('std = ', train_std, '\nmean ', train_mean)

    # %% Model and optimizer

    autoencoder = AutoEncoder(strides=STRIDES, bottleneck_channel_size=BOTTLENECK_CHANNEL_SIZE,
                              num_packets=NUM_PACKETS).to(DEVICE)

    criterion = AutoEncoderLoss()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    # %% Training
    print("\nTraining started...")

    encoder_loss_array = []
    decoder_loss_array = []

    train_acc_array = []
    val_acc_array = []

    print("\n{:<20} {:<20} {:<20} {:<20}".format('Epoch', 'train accuracy', 'validation accuracy', 'encoder loss'))

    for epoch in tqdm(range(NUM_EPOCHS)):
        encoder_running_loss = 0
        decoder_running_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            train_audio, train_messages, train_messages_reshaped = data[0].to(DEVICE, dtype=torch.float), data[1].to(
                DEVICE, dtype=torch.float), data[2].to(DEVICE, dtype=torch.float)

            optimizer.zero_grad()

            reconstructed_message, modified_audio = autoencoder(train_audio.unsqueeze(1), train_messages_reshaped)
            loss, encoder_loss, decoder_loss = criterion(modified_audio,
                                                         train_audio.unsqueeze(1),
                                                         reconstructed_message,
                                                         train_messages)
            loss.backward()

            optimizer.step()

            encoder_running_loss += encoder_loss.item()
            decoder_running_loss += decoder_loss.item()

        encoder_loss_array.append(encoder_running_loss)
        decoder_loss_array.append(decoder_running_loss)

        with torch.no_grad():
            validation_dataloader = DataLoader(validation_set, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)
            train_test_dataloader = DataLoader(train_set, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)

            val_acc = calc_accuracy(autoencoder, validation_dataloader, packet_len=PACKET_LEN, device=DEVICE)
            val_acc_array.append(val_acc)

            train_acc = calc_accuracy(autoencoder, train_test_dataloader, packet_len=PACKET_LEN, device=DEVICE)
            train_acc_array.append(train_acc)

        print("\ne: {:<20}ta: {:<20.2f}va: {:<20.2f} {:<20.3f}".format(epoch, train_acc, val_acc, encoder_running_loss))

    # %% Plot
    plt.figure()
    plt.title("Loss normalized")
    plt.xlabel("epoch [num]")
    plt.ylabel("loss [num]")
    plt.plot(encoder_loss_array / np.max(encoder_loss_array), label='encoder')
    plt.plot(decoder_loss_array / np.max(decoder_loss_array), label='decoder')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Accuracy")
    plt.xlabel("epoch [num]")
    plt.ylabel("accuracy [0..1]")
    plt.plot(train_acc_array, label='train')
    plt.plot(val_acc_array, label='validation')
    plt.legend()
    plt.show()

    # %% Test set
    with torch.no_grad():
        test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

        test_acc = calc_accuracy(autoencoder, test_dataloader, packet_len=PACKET_LEN, device=DEVICE)
        print('\nTest accuracy is {:2.2f} %'.format(100 * test_acc))

    print('\nSaving data...')

    Path(SAVE_MODELS_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    Path(ORIGINAL_AUDIO_PATH).mkdir(parents=True, exist_ok=True)
    Path(STEGANOGRAPHIC_AUDIO_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_PARAMETERS_PATH).mkdir(parents=True, exist_ok=True)

    torch.save(autoencoder.state_dict(),
               os.path.join(MODEL_PATH, MODEL_NAME + MODEL_EXTENSION))

    params = {
        'NUM_PACKETS': NUM_PACKETS,
        'PACKET_LEN': PACKET_LEN,
        'BOTTLENECK_CHANNEL_SIZE': BOTTLENECK_CHANNEL_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'STRIDES': STRIDES
    }
    save_parameters(os.path.join(MODEL_PARAMETERS_PATH, TRAINING_PARAMETERS_JSON), params)

    # np.save(os.path.join(MODEL_PARAMETERS_PATH, 'strides.npy'), STRIDES)

    with torch.no_grad():
        wav_saving_dataloader = DataLoader(test_set, batch_size=WAV_SAVING_NUM, shuffle=True)
        _, _, original_audio, modified_audio = pass_data_through(autoencoder, wav_saving_dataloader, DEVICE)

        original_audio = original_audio.squeeze()
        modified_audio = modified_audio.squeeze()

    for i in range(WAV_SAVING_NUM):
        write(os.path.join(ORIGINAL_AUDIO_PATH, 'sample' + str(i) + '.wav'), FS, original_audio[i, :])
        write(os.path.join(STEGANOGRAPHIC_AUDIO_PATH, 'sample' + str(i) + '.wav'), FS, modified_audio[i, :])

    print('Done')

    # TODO Save some results like accuracy maybe, in a csv file
