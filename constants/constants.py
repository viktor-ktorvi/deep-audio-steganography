import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HOLDOUT_RATIO = 0.8
SIGNAL_LEN = 16384

CHANNELS = {
    "large": 25,
    "medium": 15,
    "small": 5
}

KERNELS = {
    "large": 65,
    "medium": 33,
    "small": 5
}