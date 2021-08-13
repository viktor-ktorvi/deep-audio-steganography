import torch
from torch.utils.data import DataLoader
import numpy as np

from data_loading import get_dataset
from constants import DEVICE
from parameters import BATCH_SIZE

if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    train_set, validation_set, test_set = get_dataset()

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    train_std, train_mean = torch.std_mean(train_set.dataset.tensors[0], unbiased=False)
    print('std = ', train_std, '\nmean ', train_mean)

