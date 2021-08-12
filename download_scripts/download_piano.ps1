Write-Host "Downloading piano"
wget https://s3.amazonaws.com/wavegan-v1/models/piano.ckpt.index -O piano\model.ckpt.index
Write-Host "1/3"
wget https://s3.amazonaws.com/wavegan-v1/models/piano.ckpt.data-00000-of-00001 -O piano\model.ckpt.data-00000-of-00001
Write-Host "2/3"
wget https://s3.amazonaws.com/wavegan-v1/models/piano_infer.meta -O piano\infer.meta
Write-Host "3/3"
Write-Host "Piano downloaded"
