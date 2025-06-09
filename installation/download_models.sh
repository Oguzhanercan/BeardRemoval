source /home/oguzhan/Desktop/TBFR/venv/bin/activate

#bash ./download_hf.sh black-forest-labs/FLUX.1-dev ../models/FLUX.1-dev
#bash ./download_hf.sh sayakpaul/flux.1-dev-nf4-with-bnb-integration ../models/4bit_transformer

#bash ./download_zip.sh "https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliucz_connect_ust_hk/EWlGF0WfawJIrJ1Hn85_-3gB0MtwImAnYeWXuleVQcukMg?e=Gcjugg&download=1" ../models/
mv ../models/models/* ../models/
rm ../models/models 

#wget https://github.com/prashants975/Beard-Hair-Image-Segmentation/edit/main/models/best_hair_117_epoch_v4.pt

#mv best_hair_117_epoch_v4.pt ../models/best_hair_117_epoch_v4.pt ../models/