Write-Host "Downloading birds"
wget https://s3.amazonaws.com/wavegan-v1/models/birds.ckpt.index -O birds\model.ckpt.index
Write-Host "1/3"
wget https://s3.amazonaws.com/wavegan-v1/models/birds.ckpt.data-00000-of-00001 -O birds\model.ckpt.data-00000-of-00001
Write-Host "2/3"
wget https://s3.amazonaws.com/wavegan-v1/models/birds_infer.meta -O birds\infer.meta
Write-Host "3/3"
Write-Host "Birds downloaded"