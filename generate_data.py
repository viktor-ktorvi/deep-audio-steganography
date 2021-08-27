import tensorflow as tf
import os
import numpy as np
import time as time
from scipy.io.wavfile import write
from pathlib import Path
from tqdm import tqdm

from constants.paths import DATA_FILENAME, PRETRAINED_MODELS_PATH, TRAIN_DATA_PATH, INFERENCE_DATA_FOLDER, \
    MIXED_DATA_FOLDER
from constants.constants import SIGNAL_LEN, FS

DATASET = 'birds'  # one of 'digits', 'speech', 'birds', 'drums', 'piano'

DATA_PATH = TRAIN_DATA_PATH

GENERATE_BATCH_SIZE = 64
NUM_BATCHES = 75
INPUT_LEN = 100
OUTPUT_LEN = SIGNAL_LEN
SAVE_AS_WAV_PERCENT = 0.01

Fs = FS  # Hz

'''
    Generate audio data(as .npy and .wav) using the pretrained WaveGAN 
'''

if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(PRETRAINED_MODELS_PATH, DATASET, 'infer.meta'))
    graph = tf.compat.v1.get_default_graph()
    sess = tf.compat.v1.InteractiveSession()
    saver.restore(sess, os.path.join(PRETRAINED_MODELS_PATH, DATASET, 'model.ckpt'))

    data = np.zeros((NUM_BATCHES * GENERATE_BATCH_SIZE, OUTPUT_LEN), dtype=np.float32)
    noise = np.zeros((NUM_BATCHES * GENERATE_BATCH_SIZE, INPUT_LEN), dtype=np.float32)

    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')

    start = time.time()
    for i in tqdm(range(NUM_BATCHES)):
        _z = np.random.randn(GENERATE_BATCH_SIZE, INPUT_LEN)

        # G_z_spec is not being used
        _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})

        data[i * GENERATE_BATCH_SIZE:(i + 1) * GENERATE_BATCH_SIZE, :] = _G_z
        noise[i * GENERATE_BATCH_SIZE:(i + 1) * GENERATE_BATCH_SIZE, :] = _z

    print('Finished! (Took {} seconds)'.format(time.time() - start))

    # %% Saving audio
    print('\nSaving data...')
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(DATA_PATH, DATASET)).mkdir(parents=True, exist_ok=True)

    # save as .npy
    with open(os.path.join(DATA_PATH, DATASET, DATA_FILENAME + '.npy'), 'wb') as f:
        np.save(f, data)

    # save every modulus_num-th sample as .wav
    modulus_num = np.ceil(1 / SAVE_AS_WAV_PERCENT) if SAVE_AS_WAV_PERCENT > 0.0 else data.shape[0]

    for i in range(data.shape[0]):
        if i % modulus_num == 0:
            write(os.path.join(DATA_PATH, DATASET, DATASET + str(i) + '.wav'), Fs, data[i, :])

    print('\nDone')
