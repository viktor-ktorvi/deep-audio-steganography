# Confirm GPU is running
from tensorflow.python.client import device_lib
import torch


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def print_dashes(num_dash):
    for i in range(num_dash):
        print('=', end='')
    print('\n')


if __name__ == "__main__":
    num_dash = 50
    print("Tensorflow\n")

    print_dashes(num_dash)

    local_devices = device_lib.list_local_devices()
    print(local_devices)

    if len(get_available_gpus()) == 0:
        for i in range(4):
            print('WARNING: Not running on a GPU! See above for faster generation')

    print('\nPytorch\n')
    print_dashes(num_dash)

    print('torch.cuda.is_available() ', torch.cuda.is_available())
    print('torch.cuda.current_device() ', torch.cuda.current_device())
    print('torch.cuda.device(0) ', torch.cuda.device(0))
    print('torch.cuda.device_count() ', torch.cuda.device_count())
    print('torch.cuda.get_device_name(0) ', torch.cuda.get_device_name(0))
