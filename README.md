# Keras implementation of Deeplabv3+
DeepLab is a state-of-art deep learning model for semantic image segmentation. 
Model is based on the original TF frozen graph. It is possible to load pretrained weights into this model. Weights are directly imported from original TF checkpoint.  

Segmentation results of original TF model. __Output Stride = 8__
<p align="center">
    <img src="imgs/seg_results1.png" width=600></br>
    <img src="imgs/seg_results2.png" width=600></br>
    <img src="imgs/seg_results3.png" width=600></br>
</p>

This result is obtained as an argmax applied to logits at exit layer   
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

How to use this model with custom input shape and custom number of classes:  
`from model import Deeplabv3`  
`deeplab_model = Deeplabv3(input_shape=(512,512,3), classes=4, weights='pascal_voc', OS=8)`   

After that you will get a usual Keras model which you can train using .fit and .fit_generator methods

I'm using   
Keras==2.1.5  
tensorflow-gpu==1.6.0  
CUDA==9.0   
