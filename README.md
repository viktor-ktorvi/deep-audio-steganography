# Deep audio steganography

## Quick start

### Download the WaveGAN models
Download the pretrained WaveGAN Generator as a .ckpt file by running _download_wavegan.py_ but before that, make sure you are on 
Windows and that you have running PowerShell commands enabled. 
See [link](https://superuser.com/questions/106360/how-to-enable-execution-of-powershell-scripts) on how to do that. If 
you're on Linux you should be able to just run the _wget_ commands in the terminal (I can write a shell script but I 
can't test it :disappointed_relieved:). The folder _download_scritps_ has PowerShell scripts ready for use 
but _download_wavegan.py_ just runs the same commands from a subrpocess call.

### Generating data with WaveGAN
Choose you're _DATASET_ and run _generate_data.py_