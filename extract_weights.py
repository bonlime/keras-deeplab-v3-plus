from __future__ import print_function

import os
import re
from glob import glob
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file


def get_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.
    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('xception_65_', '')
    filename = filename.replace('decoder_','',1)
    filename = filename.replace('BatchNorm','BN')
    if 'Momentum' in filename:
        return None
    if 'entry_flow' in filename or 'exit_flow' in filename:
        filename = filename.replace('_unit_1_xception_module','')
    elif 'middle_flow' in filename:
        filename = filename.replace('_block1','')
        filename = filename.replace('_xception_module','')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'

def extract_tensors_from_checkpoint_file(filename, output_folder='weights'):
    """Extract tensors from a TF checkpoint file.
    # Arguments
        filename: TF checkpoint file
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # convert tensor name into the corresponding Keras layer weight name and save
        filename = get_filename(key)
        if filename:
            path = os.path.join(output_folder, get_filename(key))
            arr = reader.get_tensor(key)
            np.save(path, arr)
            print("tensor_name: ", key)

CKPT_URL = 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
MODEL_DIR = 'models'
MODEL_SUBDIR = 'deeplabv3_pascal_trainval'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
checkpoint_tar = get_file(
    'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    CKPT_URL,
    extract=True,
    cache_subdir='',
    cache_dir=MODEL_DIR)

checkpoint_file = os.path.join(MODEL_DIR,MODEL_SUBDIR, 'model.ckpt')
extract_tensors_from_checkpoint_file(checkpoint_file)
