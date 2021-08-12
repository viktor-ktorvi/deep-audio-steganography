Write-Host "Downloading digits"
wget https://s3.amazonaws.com/wavegan-v1/models/sc09.ckpt.index -O digits\model.ckpt.index
Write-Host "1/3"
wget https://s3.amazonaws.com/wavegan-v1/models/sc09.ckpt.data-00000-of-00001 -O digits\model.ckpt.data-00000-of-00001
Write-Host "2/3"
wget https://s3.amazonaws.com/wavegan-v1/models/sc09_infer.meta -O digits\infer.meta
Write-Host "3/3"
Write-Host "Digits downloaded"
