import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from data_loading import get_dataset
from constants.parameters import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from constants.constants import DEVICE

from network_modules.autoencoder import AutoEncoder
from loss.autoencoder_loss import AutoEncoderLoss

STRIDES = [4, 8, 8]

if __name__ == '__main__':
    # %% Seeds
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # %% Loading the data
    train_set, validation_set, test_set = get_dataset()
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    train_std, train_mean = torch.std_mean(train_set.dataset.tensors[0], unbiased=False)
    print('std = ', train_std, '\nmean ', train_mean)

    # %% Model and optimizer

    autoencoder = AutoEncoder(strides=STRIDES).to(DEVICE)

    criterion = AutoEncoderLoss()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    # %% Training
    print("Training started")

    encoder_loss_array = []
    decoder_loss_array = []

    train_acc_array = []
    val_acc_array = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        encoder_running_loss = 0
        decoder_running_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            train_audio, train_messages, train_messages_reshaped = data[0].to(DEVICE, dtype=torch.float), data[1].to(
                DEVICE, dtype=torch.float), data[2].to(DEVICE, dtype=torch.float)

            optimizer.zero_grad()

            reconstructed_message, modified_audio = autoencoder(train_audio, train_messages_reshaped)
            loss, encoder_loss, decoder_loss = AutoEncoderLoss(modified_audio, train_audio, reconstructed_message,
                                                               train_messages)
            loss.backward()

            optimizer.step()

            encoder_running_loss += encoder_loss.item()
            decoder_running_loss += decoder_loss.item()

        encoder_loss_array.append(encoder_running_loss)
        decoder_loss_array.append(decoder_running_loss)

        # %% Plot
        plt.figure()
        plt.title("Loss normalized")
        plt.xlabel("epoch [num]")
        plt.ylabel("loss [num]")
        plt.plot(encoder_loss_array / np.max(encoder_loss_array), label='encoder')
        plt.plot(decoder_loss_array / np.max(decoder_loss_array), label='decoder')
        plt.legend()
        plt.show()
