# keras-deeplab-v3-plus
Keras implementation of Deeplab v.3
Model is based on the original TF frozen graph. It is possible to load pretrained weights into this model. Weights are directly imported from original TF checkpoint  

Segmentation results of original TF model. __Output Stride = 8__
<p align="center">
    <img src="imgs/seg_results1.png" width=600></br>
    <img src="imgs/seg_results2.png" width=600></br>
    <img src="imgs/seg_results3.png" width=600></br>
</p>

Segmentation results of this repo model with loaded weights:  
__This results were obtained with Output Stride = 16__  
Right now I cant make it work with OS = 8. Results qualiy is much worse than original 
<p align="center">
    <img src="imgs/my_seg_results1.png" width=600></br>
    <img src="imgs/my_seg_results2.png" width=600></br>
    <img src="imgs/my_seg_results3.png" width=600></br>
</p>
