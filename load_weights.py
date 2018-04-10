from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from keras.models import Model
from model import Deeplabv3


WEIGHTS_DIR = 'weights'
MODEL_DIR = 'models'
OUTPUT_WEIGHT_FILENAME = 'deeplabv3_weights_tf_dim_ordering_tf_kernels.h5'


print('Instantiating an empty Deeplabv3+ model...')
model = Deeplabv3(input_shape=(512, 512, 3),num_classes = 21)

WEIGHTS_DIR = 'weights/'
print('Loading weights from', WEIGHTS_DIR)

for layer in tqdm(model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
            weights.append(weight_arr)
        layer.set_weights(weights)


print('Saving model weights...')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model.save_weights(os.path.join(MODEL_DIR, OUTPUT_WEIGHT_FILENAME))
