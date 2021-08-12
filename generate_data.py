import tensorflow as tf
import os
import numpy as np
import time as time
from scipy.io.wavfile import write
from pathlib import Path
from tqdm import tqdm

DATASET = 'birds'  # one of 'digits', 'speech', 'birds', 'drums', 'piano'

DATA_PATH = 'data'
DATA_FILENAME = 'audio_data'
MODELS_PATH = 'waveGAN_models'

BATCH_SIZE = 64
NUM_BATCHES = 100
INPUT_LEN = 100
OUTPUT_LEN = 16384

Fs = 16000  # Hz

if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(MODELS_PATH, DATASET, 'infer.meta'))
    graph = tf.compat.v1.get_default_graph()
    sess = tf.compat.v1.InteractiveSession()
    saver.restore(sess, os.path.join(MODELS_PATH, DATASET, 'model.ckpt'))

    data = np.zeros((NUM_BATCHES * BATCH_SIZE, OUTPUT_LEN), dtype=np.float32)
    noise = np.zeros((NUM_BATCHES * BATCH_SIZE, INPUT_LEN), dtype=np.float32)

    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')

    start = time.time()
    for i in tqdm(range(NUM_BATCHES)):
        _z = np.random.randn(BATCH_SIZE, INPUT_LEN)

        # G_z_spec is not being used
        _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})

        data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] = _G_z
        noise[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] = _z

    print('Finished! (Took {} seconds)'.format(time.time() - start))

    # %% Saving audio
    print('\nSaving data...')
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(DATA_PATH, DATASET)).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(DATA_PATH, DATASET, DATA_FILENAME + '.npy'), 'wb') as f:
        np.save(f, data)

    for i in range(data.shape[0]):
        if i % round(data.shape[0] * 0.05) == 0:
            write(os.path.join(DATA_PATH, DATASET, DATASET + str(i) + '.wav'), Fs, data[i, :])

    print('\nDone')