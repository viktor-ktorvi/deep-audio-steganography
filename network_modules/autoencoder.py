import numpy as np
import torch
from torch import nn

from network_modules.encoder import Encoder
from network_modules.decoder import Decoder

from data_loading import reshape_messages
from constants.constants import CHANNELS, KERNELS, SIGNAL_LEN, DEVICE
from constants.parameters import MESSAGE_LEN, BOTTLENECK_CHANNEL_SIZE


class AutoEncoder(nn.Module):
    def __init__(self, strides):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(strides=strides)
        self.decoder = Decoder()

    def forward(self, x, message_reshaped):
        modified_audio = self.encoder(x, message_reshaped)
        reconstructed_message = self.decoder(modified_audio)

        return reconstructed_message, modified_audio


if __name__ == '__main__':
    autoencoder = AutoEncoder(strides=[4, 8, 8]).to(DEVICE)
    print(autoencoder)

    batch_size = 17

    messages = np.random.randint(low=0, high=2, size=(batch_size, MESSAGE_LEN))
    messages_reshaped = reshape_messages(messages)
    messages_reshaped_tensor = torch.tensor(messages_reshaped).to(DEVICE)

    x = torch.randn(size=(batch_size, 1, SIGNAL_LEN)).to(DEVICE)
    reconstructed_message, modified_audio = autoencoder(x, messages_reshaped_tensor)
