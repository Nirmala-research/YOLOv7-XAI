# Image-Reconstruction Technique
  Medical Image Reconstruction - Dermascopic images

## Overview
  An official implementation of Single image reconstruction technique for Melanoma Skin lesion images for feature extraction using PyTorch.

## Requirments
  Matlab R2019
  
  Python 3.10.10
  
  PyTorch 1.4
  
  Pillow 5.1.0
  
  scikit-image 0.19.3
  
  numpy 1.14.5
  
  This was tested on Python 3.7. To install the required packages, use the provided requirements.txt file like so:
            
```
pip install -r requirements.txt
```
            

## Datasets
  ISIC Challenge Datasets 2020  https://challenge.isic-archive.com/data/#2020
  
  PH2 Dataset https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar
## Pre-trained model
  MELLiResNet 
  
  MELLIGAN
  
  https://drive.google.com/file/d/1FV0T8C_0Z6oMUuvOq9ELs6EsXqxLuS5O/view?usp=share_link
## Input Size changing
The provided model was trained on ISIC 2019, PH2 dataset and mednode image inputs, but to run it on inputs of arbitrary size, you'll have to change the input shape as given
```
from tensorflow import keras

# Load the model
model = keras.models.load_model('models/generator.h5')

# Define arbitrary spatial dims, and 3 channels.
inputs = keras.Input((None, None, 3))

# Trace out the graph using the input:
outputs = model(inputs)

# Override the model:
model = keras.models.Model(inputs, outputs)

# Now you are free to predict on images of any size.
```
## Experimental Results
  The experimental results on the benchmark datasets.
  ### Quantitave Results
  | Algorithm|  | Bicubic | ESPCN | SRGAN | ESRGAN | MELLIGAN (MY model)|
  | ---------|--| ------- |-------|-------|--------|--------|
  |ISIC 2020| PSNR  | 23.16  | 27.52|25.85|28.54|40.12|
  | Dataset | SSIM  | 0.7244 |0.795|0.7947 |0.8145 |0.9465|
  |PH2| PSNR  |21.51|	24.6| 23.62|	24.53	|38.84|
  |  Dataset  | SSIM  |0.606|	0.7052|	0.6975|	0.6711|	0.9314|
  |Med node| PSNR  | 21.71|	22.3|	22.11|	22.79|	37.51|
  |  Dataset| SSIM  |0.6317|	0.6595|	0.6774|	0.705| 0.9215|
 ### Qualitative Results
![image](https://user-images.githubusercontent.com/107538530/218392416-8f738bfb-e019-404c-a9b8-6255387745ac.png)


## Comments
  The queries and comments on my codes can be forwarded to v.nirmalaresearch@gmail.com
## Contributors
<!-- Copy-paste in your Readme.md file -->

<a href = "https://github.com/Tanu-N-Prabhu/Python/graphs/contributors">
  <img src = "https://contrib.rocks/image?repo = GitHub_username/repository_name"/>
</a>

Made with [contributors-img](https://contrib.rocks)

 
