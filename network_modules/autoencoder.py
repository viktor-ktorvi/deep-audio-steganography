from torch import nn

from network_modules.encoder import Encoder
from network_modules.decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self, strides, bottleneck_channel_size, num_packets):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(strides=strides, bottleneck_channel_size=bottleneck_channel_size)
        self.decoder = Decoder(message_len=num_packets)

    def forward(self, x, message_reshaped):
        modified_audio = self.encoder(x, message_reshaped)
        reconstructed_message = self.decoder(modified_audio)

        return reconstructed_message, modified_audio

    def decode(self, modified_audio):
        return self.decoder(modified_audio)


if __name__ == '__main__':
    pass
