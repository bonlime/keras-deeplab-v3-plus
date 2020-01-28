# Keras implementation of Deeplabv3+  
**This repo is not longer maintained. I won't respond to issues but will merge PR**  
DeepLab is a state-of-art deep learning model for semantic image segmentation.  

Model is based on the original TF frozen graph. It is possible to load pretrained weights into this model. Weights are directly imported from original TF checkpoint.  

Segmentation results of original TF model. __Output Stride = 8__
<p align="center">
    <img src="imgs/seg_results1.png" width=600></br>
    <img src="imgs/seg_results2.png" width=600></br>
    <img src="imgs/seg_results3.png" width=600></br>
</p>

Segmentation results of this repo model with loaded weights and __OS = 8__  
Results are identical to the TF model  
<p align="center">
    <img src="imgs/my_seg_results1_OS8.png" width=600></br>
    <img src="imgs/my_seg_results2_OS8.png" width=600></br>
    <img src="imgs/my_seg_results3_OS8.png" width=600></br>
</p>

Segmentation results of this repo model with loaded weights and __OS = 16__  
Results are still good
<p align="center">
    <img src="imgs/my_seg_results1_OS16.png" width=600></br>
    <img src="imgs/my_seg_results2_OS16.png" width=600></br>
    <img src="imgs/my_seg_results3_OS16.png" width=600></br>
</p>

### How to get labels
Model will return tensor of shape `(batch_size, height, width, num_classes)`. To obtain labels, you need to apply argmax to logits at exit layer. Example of predicting on `image1.jpg`:  

```python
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from model import Deeplabv3

# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

trained_image_width=512 
mean_subtraction_value=127.5
image = np.array(Image.open('imgs/image1.jpg'))

# resize to max dimension of images from training dataset
w, h, _ = image.shape
ratio = float(trained_image_width) / np.max([w, h])
resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

# apply normalization for trained dataset images
resized_image = (resized_image / mean_subtraction_value) - 1.

# pad array to square image to match training images
pad_x = int(trained_image_width - resized_image.shape[0])
pad_y = int(trained_image_width - resized_image.shape[1])
resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

# make prediction
deeplab_model = Deeplabv3()
res = deeplab_model.predict(np.expand_dims(resized_image, 0))
labels = np.argmax(res.squeeze(), -1)

# remove padding and resize back to original image
if pad_x > 0:
    labels = labels[:-pad_x]
if pad_y > 0:
    labels = labels[:, :-pad_y]
labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

plt.imshow(labels)
plt.waitforbuttonpress()
```

### How to use this model with custom input shape and custom number of classes
```python
from model import Deeplabv3
deeplab_model = Deeplabv3(input_shape=(384, 384, 3), classes=4)  
#or you can use None as shape
deeplab_model = Deeplabv3(input_shape=(None, None, 3), classes=4)
```
After that you will get a usual Keras model which you can train using `.fit` and `.fit_generator` methods.

### How to train this model

Useful parameters can be found in the [original repository](https://github.com/tensorflow/models/blob/master/research/deeplab/train.py).

Important notes:
1. This model doesn’t provide default weight decay, user needs to add it themselves.
2. Due to huge memory use with `OS=8`, Xception backbone should be trained with `OS=16` and only inferenced with `OS=8`.
3. User can freeze feature extractor for Xception backbone (first 356 layers) and only fine-tune decoder. Right now (March 2019), there is a problem with finetuning Keras models with BN. You can read more about it [here](https://github.com/keras-team/keras/pull/9965).

#### Known issues
This model can be retrained [check this notebook](https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/master/segmentation.ipynb). Finetuning is tricky and difficult because of the confusion between `training` and `trainable` in Keras. See [this issue](https://github.com/bonlime/keras-deeplab-v3-plus/issues/56) for a discussion and possible alternatives. 


### How to load model
In order to load model after using model.save() use this code:

```python
from model import relu6
deeplab_model = load_model('example.h5')
```

### Xception vs MobileNetv2
There are 2 available backbones. Xception backbone is more accurate, but has 25 times more parameters than MobileNetv2. 

For MobileNetv2 there are pretrained weights only for `alpha=1`. However, you can initiate model with different values of alpha.


### Requirement
The latest vesrion  of this repo uses TF Keras, so you only need TF 2.0+ installed  
tensorflow-gpu==2.0.0a0  
CUDA==9.0   

-------- 
If you want to use older version, use following commands:
```bash
git clone https://github.com/bonlime/keras-deeplab-v3-plus/
cd keras-deeplab-v3-plus/
git checkout 714a6b7d1a069a07547c5c08282f1a706db92e20
```
tensorflow-gpu==1.13  
Keras==2.2.4  
