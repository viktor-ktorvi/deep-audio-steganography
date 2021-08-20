import torch
from torch import nn

from utils.sizes import Conv1DLayerSizes, LinearLayerSizes
from constants.constants import CHANNELS, KERNELS, SIGNAL_LEN, DEVICE
from constants.parameters import MESSAGE_LEN


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        strides = [4, 8, 4]
        kernel_sizes = [KERNELS['large'], KERNELS['large'], KERNELS['medium']]
        out_channels = [CHANNELS['medium'], CHANNELS['large'], CHANNELS['small']]

        linear_out_features = [8 * MESSAGE_LEN, MESSAGE_LEN]
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

        self.conv = nn.Sequential(*conv_modules)

        self.linear_sizes = []
        for i in range(2):
            self.linear_sizes.append(
                LinearLayerSizes(
                    in_features=self.linear_sizes[-1].parameters["out_features"] if i != 0 else self.conv_sizes[
                                                                                                    -1].out_len *
                                                                                                self.conv_sizes[
                                                                                                    -1].parameters[
                                                                                                    "out_channels"],
                    out_features=linear_out_features[i]
                )
            )

        linear_modules = []
        for i, linear_size in enumerate(self.linear_sizes):
            if i != len(self.linear_sizes) - 1:
                linear_modules.append(nn.Sequential(
                    nn.Linear(**linear_size.parameters),
                    # nn.BatchNorm1d(linear_size.parameters['out_features']),
                    nn.ReLU()
                ))
            else:
                linear_modules.append(nn.Sequential(
                    nn.Linear(**linear_size.parameters),
                    # nn.Tanh()
                ))

        self.dense = nn.Sequential(*linear_modules)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.linear_sizes[0].parameters["in_features"])
        x = self.dense(x)
        return x


if __name__ == '__main__':
    decoder = Decoder().to(DEVICE)
    print(decoder)

    batch_size = 17

    x = torch.randn(size=(batch_size, 1, SIGNAL_LEN)).to(DEVICE)
    y = decoder(x)
