Write-Host "Downloading drums"
wget https://s3.amazonaws.com/wavegan-v1/models/drums.ckpt.index -O drums\model.ckpt.index
Write-Host "1/3"
wget https://s3.amazonaws.com/wavegan-v1/models/drums.ckpt.data-00000-of-00001 -O drums\model.ckpt.data-00000-of-00001
Write-Host "2/3"
wget https://s3.amazonaws.com/wavegan-v1/models/drums_infer.meta -O drums\infer.meta
Write-Host "3/3"
Write-Host "Drums downloaded"