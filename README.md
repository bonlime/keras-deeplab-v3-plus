# keras-deeplab-v3-plus
Keras implementation of Deeplab v.3
Model is based on the original TF frozen graph. It is possible to load pretrained weights into this model. Weights are directly imported from original TF checkpoint  

Segmentation results of original TF model. __Output Stride = 8__
<p align="center">
    <img src="imgs/seg_results1.png" width=600></br>
    <img src="imgs/seg_results2.png" width=600></br>
    <img src="imgs/seg_results3.png" width=600></br>
</p>
This result is obtained as an argmax applied to logits at exit layer   
Segmentation results of this repo model with loaded weights.  

This results were obtained with Output Stride = 16

<p align="center">
    <img src="imgs/my_seg_results1.png" width=600></br>
    <img src="imgs/my_seg_results2.png" width=600></br>
    <img src="imgs/my_seg_results3.png" width=600></br>
</p>

Segmentation results of this repo model with OS = 8  
I changed dilation (atrous) rate in ASPP branches to [12,24,36] and changed stride in entry_block_1 to 1 
Results qualiy is much worse than original  
<p align="center">
    <img src="imgs/my_seg_results1_OS8.png" width=600></br>
    <img src="imgs/my_seg_results2_OS8.png" width=600></br>
    <img src="imgs/my_seg_results3_OS8.png" width=600></br>
</p>
