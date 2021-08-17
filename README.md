# Deep audio steganography

## Quick start

### Requirements

Lots of them. 
* Firstly, to download the WaveGAN generator model you should be on Windows because I call PowerShell 
commands, but you can easily work around this if you're on Linux by either writing a shell script or downloading the
models manually and  putting them in the _waveGAN_models_ folder.

* Secondly, install Tensorflow. I think any version above 2.0 should work but I installed 2.5. To install TF along with 
cuda and cuDNN follow this [tutorial](https://www.youtube.com/watch?v=hHWkvEcDBO0) (fantastic channel). TF is only
used to initially generate data.

* Thirdly, install PyTorch. I installed the version that was relevant as of august 2021.

* Finally, install everything else. I really should make "requirements.txt" file. All the other libraries are standard.

### Download the WaveGAN models
Download the pretrained WaveGAN Generator as a .ckpt file by running _download_wavegan.py_ but before that, make sure you are on 
Windows and that you have running PowerShell commands enabled. 
See [link](https://superuser.com/questions/106360/how-to-enable-execution-of-powershell-scripts) on how to do that. If 
you're on Linux you should be able to just run the _wget_ commands in the terminal (I can write a shell script but I 
can't test it :disappointed_relieved:). The folder _download_scritps_ has PowerShell scripts ready for use 
but _download_wavegan.py_ just runs the same commands from a subrpocess call.

### Generating data with WaveGAN
Choose you're _DATASET_ and run _generate_data.py_

### Train your model
Train your model with the _train.py_ script