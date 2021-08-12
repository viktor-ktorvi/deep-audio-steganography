import subprocess
import sys
from pathlib import Path
import os

# one or more of 'digits', 'speech', 'birds', 'drums', 'piano'
datasets = ['digits', 'speech', 'birds', 'drums', 'piano']

download_filename = {
    'birds': 'birds',
    'digits': 'sc09',
    'drums': 'drums',
    'piano': 'piano',
    'speech': 'timit'
}

BASE_URL = 'https://s3.amazonaws.com/wavegan-v1/models/'

'''
    Download the pretrained models of WaveGAN as .ckpt files
'''

if __name__ == "__main__":

    # create a folder for the models
    models_dir = 'waveGAN_models'
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    os.chdir(models_dir)

    for dataset in datasets:
        # create folder for model, haven't tested this
        Path(dataset).mkdir(parents=True, exist_ok=True)
        print("Downloading", dataset, "...")

        '''
            Download: 
            1) .ckpt.index
            2) .ckpt.data
            3) .infet.meta
        '''

        commands = [
            'wget ' + BASE_URL + download_filename[dataset] + '.ckpt.index -O ' + dataset + '\model.ckpt.index',  # 1)
            'wget ' + BASE_URL + download_filename[dataset] + '.ckpt.data-00000-of-00001 -O ' +
            dataset + '\model.ckpt.data-00000-of-00001',  # 2)
            'wget ' + BASE_URL + download_filename[dataset] + '_infer.meta -O ' + dataset + '\infer.meta'
        ]

        # run the commads in PowerShell
        for command in commands:
            p = subprocess.call(['powershell.exe', command], stdout=sys.stdout)
    print("\nDone")
