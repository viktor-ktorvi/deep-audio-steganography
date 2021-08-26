import torch
from torch import nn
import numpy as np

from utils.data_loading import reshape_messages
from utils.sizes import Conv1DLayerSizes, TransposeConv1DLayerSizes
from constants.constants import CHANNELS, KERNELS, SIGNAL_LEN, DEVICE
from constants.parameters import MESSAGE_LEN


class Encoder(nn.Module):
    def __init__(self, strides, bottleneck_channel_size):
        super(Encoder, self).__init__()

        kernel_sizes = [KERNELS['large'], KERNELS['medium'], KERNELS['small']]
        out_channels = [CHANNELS['small'], CHANNELS['medium'], bottleneck_channel_size]

        n = len(out_channels)

        self.conv_sizes = []
        for i in range(n):
            self.conv_sizes.append(
                Conv1DLayerSizes(
                    in_channels=self.conv_sizes[-1].parameters['out_channels'] if i != 0 else 1,
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    in_len=self.conv_sizes[-1].out_len if i != 0 else SIGNAL_LEN
                )
            )

        conv_modules = []
        for conv_size in self.conv_sizes:
            conv_modules.append(nn.Sequential(
                nn.Conv1d(**conv_size.parameters),
                nn.ReLU()
            ))

        self.conv1 = conv_modules[0]
        self.conv2 = conv_modules[1]
        self.conv3 = conv_modules[2]

        self.transpose_conv_sizes = []
        for i in range(n):
            self.transpose_conv_sizes.append(
                TransposeConv1DLayerSizes(
                    in_channels=self.transpose_conv_sizes[-1].parameters['out_channels'] if i != 0 else out_channels[
                                                                                                            -1] * 2,
                    out_channels=out_channels[n - 2 - i] if i != n - 1 else 1,
                    kernel_size=kernel_sizes[n - 1 - i],
                    stride=strides[n - 1 - i],
                    output_padding=strides[n - 1 - i] - 1,
                    in_len=self.transpose_conv_sizes[-1].out_len if i != 0 else self.conv_sizes[-1].out_len
                )
            )

        transpose_conv_modules = []
        for transpose_conv_size in self.transpose_conv_sizes:
            if transpose_conv_size.parameters['out_channels'] == 1:
                transpose_conv_modules.append(nn.Sequential(
                    nn.ConvTranspose1d(**transpose_conv_size.parameters),
                    nn.Tanh()
                ))
            else:
                transpose_conv_modules.append(nn.Sequential(
                    nn.ConvTranspose1d(**transpose_conv_size.parameters),
                    nn.ReLU()
                ))

        self.transpose_conv3 = transpose_conv_modules[0]
        self.transpose_conv2 = transpose_conv_modules[1]
        self.transpose_conv1 = transpose_conv_modules[2]

    def forward(self, x, messages_reshaped):
        skip1 = x
        x = self.conv1(x)

        skip2 = x
        x = self.conv2(x)

        skip3 = x
        x = self.conv3(x)

        x = torch.cat((x, messages_reshaped), axis=1)

        x = self.transpose_conv3(x) + skip3
        x = self.transpose_conv2(x) + skip2
        x = self.transpose_conv1(x) + skip1

        return x


if __name__ == '__main__':
    encoder = Encoder(strides=[4, 8, 8]).to(DEVICE)
    print(encoder)

    batch_size = 17

    messages = np.random.randint(low=0, high=2, size=(batch_size, MESSAGE_LEN))
    messages_reshaped = reshape_messages(messages)
    messages_reshaped_tensor = torch.tensor(messages_reshaped).to(DEVICE)

    x = torch.randn(size=(batch_size, 1, SIGNAL_LEN)).to(DEVICE)
    y = encoder(x, messages_reshaped_tensor)
