from sklearn import metrics
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model,Sequential
from keras import optimizers
from keras.regularizers import l2
from keras.applications import resnet50
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config import *
from math import ceil
import json
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from collections import Counter
from keras import backend as K
from keras.layers import *
from keras.models import Model, load_model
import string
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
from keras.optimizers import Adam

# global constants
DIM_ORDERING = 'tf'
CONCAT_AXIS = -1
def inception_module(x, params, dim_ordering, concat_axis,
                     subsample=(1, 1), activation='relu',
                     border_mode='same', weight_decay=None):

    # https://gist.github.com/nervanazoo/2e5be01095e935e90dd8  #
    # file-googlenet_neon-py

    (branch1, branch2, branch3, branch4) = params

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    pathway1 = Convolution2D(branch1[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)

    pathway2 = Convolution2D(branch2[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)
    pathway2 = Convolution2D(branch2[1], 3, 3,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway2)

    pathway3 = Convolution2D(branch3[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)
    pathway3 = Convolution2D(branch3[1], 5, 5,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway3)

    pathway4 = MaxPooling2D(pool_size=(1, 1), dim_ordering=DIM_ORDERING)(x)
    pathway4 = Convolution2D(branch4[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway4)

    return Concatenate()([pathway1, pathway2, pathway3, pathway4])


def conv_layer(x, nb_filter, nb_row, nb_col, dim_ordering,
               subsample=(1, 1), activation='relu',
               border_mode='same', weight_decay=None, padding=None):

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      bias=False,
                      dim_ordering=dim_ordering)(x)

    if padding:
        for i in range(padding):
            x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x


def GoogLeNet(input_shape):
    img_input = Input(input_shape)
    x = conv_layer(img_input, nb_col=7, nb_filter=64,
                   nb_row=7, dim_ordering=DIM_ORDERING, padding=3)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)

    x = conv_layer(x, nb_col=1, nb_filter=64,
                   nb_row=1, dim_ordering=DIM_ORDERING)
    x = conv_layer(x, nb_col=3, nb_filter=192,
                   nb_row=3, dim_ordering=DIM_ORDERING, padding=1)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)

    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    # AUX 1 - Branch HERE
    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    # AUX 2 - Branch HERE
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = AveragePooling2D(strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Flatten()(x)

    return Model(img_input,x)



def matchnet(input_shape):
    input1 = Input(input_shape) 
    x = Conv2D(24, kernel_size=(7, 7),activation='relu')(input1)

    # pool0
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # conv1
    x = Conv2D(64, kernel_size=(5, 5),
                activation='relu')(x)
    # pool1
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # conv2
    x = Conv2D(96, kernel_size=(3, 3),
                activation='relu')(x)
    # conv3
    x = Conv2D(96, kernel_size=(3, 3),
                activation='relu')(x)
    # conv4
    x = Conv2D(64, kernel_size=(3, 3),
                activation='relu')(x)
    # pool4
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # bottleneck
    x = Flatten()(x)
    return Model(input1,x)

def lenet5(input_shape):
    input1 = Input(input_shape) 
    x = Conv2D(20, [5, 5], padding='same', activation='relu')(input1)

    x = MaxPooling2D((2, 2), strides=2)(x)

    x = Conv2D(50, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    x = Flatten()(x)
    return Model(input1,x)

def mccnn(input_shape):
    input1 = Input(input_shape) 
    x = Conv2D(112, (3,3), activation='relu')(input1)
    x = Conv2D(112, (3,3), activation='relu' )(x)
    x = Conv2D(112, (3,3), activation='relu' )(x)
    x = Conv2D(112, (3,3), activation='relu' )(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    return Model(input1,x)

def resnet6(input_shape):

    input1 = Input(input_shape) 

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(input1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    x = Flatten()(x5)

    return Model(input1,x)

def resnet8(input_shape):

    input1 = Input(input_shape)
    
    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(input1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)

    return Model(input1,x) 

def resnet50_model(input_shape):
  resnet_model = resnet50.ResNet50()
  return Model(inputs=resnet_model.input,outputs=resnet_model.get_layer("avg_pool").output)


def vgg_original (input_shape):
    input1 = Input(input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)

    return Model(input1,x)

def small_vgg_car(input_shape):
    input1 = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv1')(input1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block4_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='block5_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)

    return Model(input1,x)

def small_vgg_plate(input_shape):
    input1 = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)

    return Model(input1,x)

def small_vgg3d(input_shape):
    input1 = Input(input_shape)
    x = Conv3D(64, (1, 3, 3), activation='relu', padding='same',name='block1_conv1')(input1)

    # Block 1
    x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv3D(128, (1, 3, 3), activation='relu', padding='same',name='block2_conv1')(x)
    x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), name='block2_pool')(x)
    x = Conv3D(128, (1, 3, 3), activation='relu', padding='same',name='block2_conv2')(x)
    x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), name='block3_pool')(x)

    # Block 3
    x = Conv3D(256, (1, 3, 3), activation='relu', padding='same',name='block3_conv1')(x)
    x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), name='block4_pool')(x)

    x = Conv3D(512, (1, 3, 3), activation='relu', padding='same',name='block4_conv1')(x)
    x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), name='block5_pool')(x)

    x = Flatten()(x)

    return Model(input1,x)

#------------------------------------------------------------------------------
def get_batch_inds(batch_size, idx, N):
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 >= N:
            idx1 = N
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds
#------------------------------------------------------------------------------
def calculate_metrics(ytrue1, ypred1):
    conf = metrics.confusion_matrix(ytrue1, ypred1, [0,1])
    maxres = (conf[1,1],
              conf[0,0],
              conf[0,1],
              conf[1,0],
        metrics.precision_score(ytrue1, ypred1) * 100,
        metrics.recall_score(ytrue1, ypred1) * 100,
        metrics.f1_score(ytrue1, ypred1) * 100,
        metrics.accuracy_score(ytrue1, ypred1) * 100)
    return maxres

#------------------------------------------------------------------------------
def test_report(model_name, model, num_test_steps, test_gen):
    print("=== Evaluating model: {:s} ===".format(model_name))
    a = open("%s_inferences_output.txt" % (model_name), "w")
    ytrue, ypred = [], []
    for i in range(num_test_steps):
        X, Y, paths = next(test_gen)
        Y_ = model.predict(X)
        for y1, yreg, y2, p0, p1 in zip(Y_[0].tolist(), Y_[1].tolist(), Y['class_output'].argmax(axis=-1).tolist(), paths[0], paths[1]):
          y1_class = np.argmax(y1)
          ypred.append(y1_class)
          ytrue.append(y2)
          a.write("%s;%s;%d;%d;%f;%s\n" % (p0, p1, y2, y1_class, yreg[0], str(y1)))

    a.write('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))
    a.close()
#------------------------------------------------------------------------------
def process_load(f1, vec_size):
    _i1 = image.load_img(f1, target_size=vec_size)
    _i1 = image.img_to_array(_i1, dtype='uint8')
    return _i1

#------------------------------------------------------------------------------
def load_img(img, vec_size, vec_size2, metadata_dict):
  iplt0 = process_load(img[0][0], vec_size)
  iplt1 = process_load(img[2][0], vec_size)
  iplt2 = process_load(img[1][0], vec_size2)
  iplt3 = process_load(img[3][0], vec_size2)

  d1 = {"i0":iplt0,
        "i1":iplt1,
        "i2":iplt2,
        "i3":iplt3,
        "l":img[4],
        "p1":img[0][0],
        "p2":img[2][0],
        "c1":img[5]['color'],
        "c2":img[5]['color']
        }
  if metadata_dict is not None:
    diff = abs(np.array(metadata_dict[img[0][0]][:7]) - np.array(metadata_dict[img[2][0]][:7])).tolist()
    for i in range(len(diff)):
      diff[i] = 1 if diff[i] else 0
    d1['metadata'] = np.array(metadata_dict[img[0][0]] + metadata_dict[img[2][0]] + diff)

  return d1

#------------------------------------------------------------------------------
def generator(features, batch_size, executor, vec_size, vec_size2, type=None,metadata_dict=None, metadata_length=0, augmentation=False, with_paths=False):
  N = len(features)
  indices = np.arange(N)
  batchInds = get_batch_inds(batch_size, indices, N)

  while True:
    for inds in batchInds:
      futures = []
      _vec_size = (len(inds),) + vec_size
      b1 = np.zeros(_vec_size)
      b2 = np.zeros(_vec_size)
      _vec_size2 = (len(inds),) + vec_size2
      b3 = np.zeros(_vec_size2)
      b4 = np.zeros(_vec_size2)

      blabels = np.zeros((len(inds)))
      p1 = []
      p2 = []
      c1 = []
      c2 = []
      if metadata_length>0:
        metadata = np.zeros((len(inds),metadata_length))

      futures = [executor.submit(partial(load_img, features[index], vec_size, vec_size2, metadata_dict)) for index in inds]
      results = [future.result() for future in futures]

      for i,r in enumerate(results):
        b1[i,:,:,:] = r['i0']
        b2[i,:,:,:] = r['i1']
        blabels[i] = r['l']
        p1.append(r['p1'])
        p2.append(r['p2'])
        c1.append(r['c1'])
        c2.append(r['c2'])
        b3[i,:,:,:] = r['i2']
        b4[i,:,:,:] = r['i3']
        if metadata_length>0:
          metadata[i,:] = r['metadata']

      if augmentation:
        b1 = augs[0][0].augment_images(b1.astype('uint8')) / 255
        b2 = augs[1][0].augment_images(b2.astype('uint8')) / 255
        b3 = augs[2][0].augment_images(b3.astype('uint8')) / 255
        b4 = augs[3][0].augment_images(b4.astype('uint8')) / 255
      else:
        b1 = b1 / 255
        b2 = b2 / 255
        b3 = b3 / 255
        b4 = b4 / 255

      blabels2 = np.array(blabels)
      blabels = np_utils.to_categorical(blabels2, 2)
      y = {"class_output":blabels, "reg_output":blabels2}
      if type is None:
        result = [[b1, b2, b3, b4], y]
      elif type == 'plate':
        result = [[b1, b2], y]
      elif type == 'car':
        result = [[b3, b4], y]
      if metadata_length>0:
        result[0].append(metadata)
      if with_paths:
          result += [[p1,p2]]

      yield result

#------------------------------------------------------------------------------

def load_img_temporal(img, vec_size, vec_size2, tam, metadata_dict):
  iplt0 = [process_load(img[0][i], vec_size) for i in range(tam)]
  iplt1 = [process_load(img[2][i], vec_size) for i in range(tam)]
  iplt2 = [process_load(img[1][i], vec_size2) for i in range(tam)]
  iplt3 = [process_load(img[3][i], vec_size2) for i in range(tam)]

  d1 = {"i0":iplt0,
        "i1":iplt1,
        "i2":iplt2,
        "i3":iplt3,
        "l":img[4],
        "p1":str(img[0]),
        "p2":str(img[2]),
        "c1":img[5]['color'],
        "c2":img[5]['color']
        }

  d1['metadata'] = []
  for i in range(tam):
    diff = abs(np.array(metadata_dict[img[0][i]][:7]) - np.array(metadata_dict[img[2][i]][:7])).tolist()
    for j in range(len(diff)):
      diff[j] = 1 if diff[j] else 0
    d1['metadata'] += metadata_dict[img[0][i]] + metadata_dict[img[2][i]] + diff
  d1['metadata'] = np.array(d1['metadata'])
  return d1
#------------------------------------------------------------------------------
def generator_temporal(features, batch_size, executor, vec_size, vec_size2, tam, metadata_dict, metadata_length, augmentation=False, with_paths=False):
  N = len(features)
  indices = np.arange(N)
  batchInds = get_batch_inds(batch_size, indices, N)

  while True:
    for inds in batchInds:
      futures = []
      _vec_size = (len(inds),tam, ) + vec_size
      b1 = np.zeros(_vec_size)
      b2 = np.zeros(_vec_size)
      _vec_size2 = (len(inds),tam, ) + vec_size2
      b3 = np.zeros(_vec_size2)
      b4 = np.zeros(_vec_size2)

      blabels = np.zeros((len(inds)))
      p1 = []
      p2 = []
      c1 = []
      c2 = []
      metadata = np.zeros((len(inds),metadata_length))

      futures = [executor.submit(partial(load_img_temporal, features[index], vec_size, vec_size2, tam, metadata_dict)) for index in inds]
      results = [future.result() for future in futures]

      for i,r in enumerate(results):
        for j in range(tam):
          b1[i,j,:,:,:] = r['i0'][j]
          b2[i,j,:,:,:] = r['i1'][j]
          b3[i,j,:,:,:] = r['i2'][j]
          b4[i,j,:,:,:] = r['i3'][j]

        blabels[i] = r['l']
        p1.append(r['p1'])
        p2.append(r['p2'])
        c1.append(r['c1'])
        c2.append(r['c2'])
        metadata[i,:] = r['metadata']

      if augmentation:
        for j in range(tam):
          b1[:,j,:] = augs[0][j].augment_images(b1[:,j,:].astype('uint8')) / 255
          b2[:,j,:] = augs[1][j].augment_images(b2[:,j,:].astype('uint8')) / 255
          b3[:,j,:] = augs[2][j].augment_images(b3[:,j,:].astype('uint8')) / 255
          b4[:,j,:] = augs[3][j].augment_images(b4[:,j,:].astype('uint8')) / 255
      else:
        for j in range(tam):
          b1[:,j,:] = b1[:,j,:] / 255
          b2[:,j,:] = b2[:,j,:] / 255
          b3[:,j,:] = b3[:,j,:] / 255
          b4[:,j,:] = b4[:,j,:] / 255

      blabels2 = np.array(blabels)
      blabels = np_utils.to_categorical(blabels2, 2)
      y = {"class_output":blabels, "reg_output":blabels2}
      result = [[b3, b4, metadata], y]

      if with_paths:
          result += [[p1,p2]]

      yield result
