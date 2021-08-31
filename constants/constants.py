import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SIGNAL_LEN = 16384  # samples
FS = 16000  # Hz

# TODO These really should be fields in the model because of loading the model for inference.
#  if I just never change them it's ok, but if I wanna start changing them then it's a hassle...
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

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
