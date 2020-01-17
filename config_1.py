# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Lambda
import imgaug as ia
from imgaug import augmenters as iaa

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

POS = 1 #positive class
NEG = 0 #negative clas
batch_size = 128
NUM_EPOCHS = 10
nchannels=3 #number of channels
image_size_w_p = 96 #image´s width for registration plate
image_size_h_p = 48 #image´s height for registration plate
image_size_w_c = 64 #image´s width for vehicle´s shape
image_size_h_c = 64 #image´s height for vehicle´s shape

path = 'data' #path containing dataset2 and json files
folder_cross1 = '%s/dataset2/Camera1' % (path)
folder_cross2 = '%s/dataset2/Camera2' % (path)
plt_name="classes"
car_name="classes_carros"
plt_folder="*/classes"
car_folder="*/classes_carros"
ocr_file = '%s/ocr/0p.txt' % (path)
metadata_length = 35
tam_max = 3
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

augs = [[],[],[],[]]

seq_car = iaa.Sequential(
  [
      iaa.Crop(px=(0, 8)),
      iaa.Affine(
  scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        shear=(-8, 8),
  order=[0,1],
  cval=(0),
  mode='constant'),
  ],
  random_order=False
)

seq_plate = iaa.Sequential(
  [
      iaa.Affine(
  scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-5, 5),
        shear=(-16, 16),
  order=[0,1],
  cval=(0),
  mode='constant'),
      iaa.Add((-10,10)),
  ],
  random_order=False
)

for i in range(tam_max):
  augs[0].append(seq_plate)
  augs[1].append(seq_plate)
  augs[2].append(seq_car)
  augs[3].append(seq_car)
