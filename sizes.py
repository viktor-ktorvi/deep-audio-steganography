import numpy as np


class ConvLayerSizes:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, in_len=0,
                 pool_kernel=1):
        self.parameters = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding if padding is not None else int(round(kernel_size - 1) / 2),
            "dilation": dilation
        }

        self.in_len = in_len
        self.pool_kernel = pool_kernel
        self.out_len = int(self.calcOutputLen() / pool_kernel)

    def calcOutputLen(self):
        out_len = (self.in_len + 2 * self.parameters["padding"] - self.parameters["dilation"] * (
                self.parameters["kernel_size"] - 1) - 1) / self.parameters["stride"] + 1

        return int(np.trunc(out_len))


class LinearLayerSizes:
    def __init__(self, in_features, out_features):
        self.parameters = {
            "in_features": in_features,
            "out_features": out_features
        }
