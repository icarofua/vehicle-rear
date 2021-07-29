# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Lambda
import albumentations as albu
from keras_metrics import *
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
kmetrics={"class_output":['acc',f1score]}
customs_func = {"f1score": f1score}
POS = 1 #positive class
NEG = 0 #negative clas
batch_size = 128
NUM_EPOCHS = 10
nchannels=3 #number of channels
image_size_w_p = 96 #image´s width for registration plate
image_size_h_p = 48 #image´s height for registration plate
image_size_w_c = 64 #image´s width for vehicle´s shape
image_size_h_c = 64 #image´s height for vehicle´s shape
path = "data"
folder_cross1 = '%s/dataset2/Camera1' % (path)
folder_cross2 = '%s/dataset2/Camera2' % (path)
plt_name="classes"
car_name="classes_carros"
plt_folder="*/classes"
car_folder="*/classes_carros"
ocr_file = '%s/OCR/output.txt' % (path)
metadata_length = 35
tam_max = 3
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))
train_augs = [[],[],[],[],[],[]]
test_augs = [[],[],[],[]]
keys = ['Set01','Set02','Set03','Set04','Set05']

seq_car = albu.Compose(
  [
      albu.IAACropAndPad(px=(0, 8)),
      albu.IAAAffine(
        scale=(0.8, 1.2),
              shear=(-8, 8),
        order=[0,1],
        cval=(0),
        mode='constant'),
      albu.ToFloat(max_value=255)
  ],p=0.7
 )

seq_car2 = albu.Compose(
  [
      albu.IAACropAndPad(px=(0, 8)),
      albu.IAAAffine(
        scale=(0.8, 1.2),
              shear=(-8, 8),
        order=[0,1],
        cval=(0),
        mode='constant'),
  ],p=0.7
 )

seq_plate = albu.Compose(
  [
      albu.IAAAffine(
        scale=(0.8, 1.2),
        translate_percent=(-0.1, 0.1),
        rotate=(-5, 5),
        shear=(-16, 16),
        order=[0,1],
        cval=(0),
        mode='constant'),
      #albu.IAAAdd((-10,10)),
      albu.ToFloat(max_value=255)
  ],p=0.7
)

AUGMENTATIONS_TEST = albu.Compose([
    albu.ToFloat(max_value=255)
])

for i in range(tam_max):
  train_augs[0].append(seq_plate)
  train_augs[1].append(seq_plate)
  train_augs[2].append(seq_car)
  train_augs[3].append(seq_car)
  train_augs[4].append(seq_car2)
  train_augs[5].append(seq_car2)
  test_augs[0].append(AUGMENTATIONS_TEST)
  test_augs[1].append(AUGMENTATIONS_TEST)
  test_augs[2].append(AUGMENTATIONS_TEST)
  test_augs[3].append(AUGMENTATIONS_TEST)
