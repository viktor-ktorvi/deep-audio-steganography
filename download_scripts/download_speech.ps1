Write-Host "Downloading speech"
wget https://s3.amazonaws.com/wavegan-v1/models/timit.ckpt.index -O speech\model.ckpt.index
Write-Host "1/3"
wget https://s3.amazonaws.com/wavegan-v1/models/timit.ckpt.data-00000-of-00001 -O speech\model.ckpt.data-00000-of-00001
Write-Host "2/3"
wget https://s3.amazonaws.com/wavegan-v1/models/timit_infer.meta -O speech\infer.meta
Write-Host "3/3"
Write-Host "Speech downloaded"