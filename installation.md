### First create an enviroment
python3 -m venv venv
### Install required packages
pip install -r installation/requirements.txt

### then you need to download and place required model files. Due to the huge model sizes, I cannot provide direct download link. So you need to download these models from original sources.

- Flux1-dev, please download flux1-dev and put it to models folder. https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main
- Flux1-dev-nf4-quantized transformer, Please download it and put it to models folder https://huggingface.co/sayakpaul/flux.1-dev-nf4-with-bnb-integration/tree/main  
- Segmentation Model, since it is a small model, it is inside of zip file, no action required.
- MagicQuill, please download magicquill model params with
wget -O models.zip "https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliucz_connect_ust_hk/EWlGF0WfawJIrJ1Hn85_-3gB0MtwImAnYeWXuleVQcukMg?e=Gcjugg&download=1"
then extract the zip file, put the contents of zip file to models folder. (at project folder).


Overall folder structure apperance for models is can be found installation/folder_structure.png