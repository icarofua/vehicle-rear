from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config_1 import *
from math import ceil
import json
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Model, load_model
import string
import pandas as pd
from sys import argv
from custom_layers import *
from collections import Counter

#------------------------------------------------------------------------------
def load_img(img, vec_size2):
  iplt2 = process_load(img[1][0], vec_size2)
  iplt3 = process_load(img[3][0], vec_size2)

  d1 = {"i2":iplt2,
        "i3":iplt3,
        "l":img[4],
        "p1":img[0][0],
        "p2":img[2][0],
        "c1":img[5]['color'],
        "c2":img[5]['color']
        }

  return d1
#------------------------------------------------------------------------------
def generator(features, batch_size, executor, vec_size2, augmentation=False, with_paths=False):
  N = len(features)
  indices = np.arange(N)
  batchInds = get_batch_inds(batch_size, indices, N)

  while True:
    for inds in batchInds:
      futures = []
      _vec_size2 = (len(inds),) + vec_size2
      b3 = np.zeros(_vec_size2)
      b4 = np.zeros(_vec_size2)

      blabels = np.zeros((len(inds)))
      p1 = []
      p2 = []
      c1 = []
      c2 = []

      futures = [executor.submit(partial(load_img, features[index], vec_size2)) for index in inds]
      results = [future.result() for future in futures]

      for i,r in enumerate(results):
        blabels[i] = r['l']
        p1.append(r['p1'])
        p2.append(r['p2'])
        c1.append(r['c1'])
        c2.append(r['c2'])
        b3[i,:,:,:] = r['i2']
        b4[i,:,:,:] = r['i3']

      if augmentation:
        b3 = augs[2][0].augment_images(b3.astype('uint8')) / 255
        b4 = augs[3][0].augment_images(b4.astype('uint8')) / 255
      else:
        b3 = b3 / 255
        b4 = b4 / 255

      blabels2 = np.array(blabels)
      blabels = np_utils.to_categorical(blabels2, 2)
      y = {"class_output":blabels, "reg_output":blabels2}
      result = [[b3, b4, b3, b4], y]

      if with_paths:
          result += [[p1,p2]]

      yield result

#------------------------------------------------------------------------------
def siamese_model(model, input2):
  left_input_P = Input(input2)
  right_input_P = Input(input2)
  left_input_C = Input(input2)
  right_input_C = Input(input2)
  convnet_car = model(input2)
  encoded_l_C = convnet_car(left_input_C)
  encoded_r_C = convnet_car(right_input_C)
  inputs = [left_input_P, right_input_P, left_input_C, right_input_C]

  # Add the distance function to the network
  x = L1_layer([encoded_l_C, encoded_r_C])
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(512, kernel_initializer='normal',activation='relu')(x)
  x = Dropout(0.5)(x)
  predF2 = Dense(2,kernel_initializer='normal',activation='softmax', name='class_output')(x)
  regF2 = Dense(1,kernel_initializer='normal',activation='sigmoid', name='reg_output')(x)
  optimizer = Adam(0.0001)
  losses = {
     'class_output': 'binary_crossentropy',
     'reg_output': 'mean_squared_error'
  }

  lossWeights = {"class_output": 1.0, "reg_output": 1.0}

  model = Model(inputs=inputs, outputs=[predF2, regF2])
  model.compile(loss=losses, loss_weights=lossWeights,optimizer=optimizer)

  return model
#------------------------------------------------------------------------------
if __name__ == '__main__':

  data = json.load(open('dataset_1.json'))

  keys = ['Set01','Set02','Set03','Set04','Set05']

  type1 = argv[1]
  name = argv[2]

  if name == 'resnet50':
    model = resnet50_model
    image_size_h_c = 224
    image_size_w_c = 224
    batch_size = 8
  elif name == 'vgg16':
    model = vgg_original
    image_size_h_c = 224
    image_size_w_c = 224
    batch_size = 8
  elif name == 'resnet8':
    model = resnet8
    image_size_h_c = 128
    image_size_w_c = 128
  elif name == 'resnet6':
    model = resnet6
    image_size_h_c = 128
    image_size_w_c = 128
  elif name == 'lenet5':
    model = lenet5
  elif name == 'matchnet':
    model = matchnet
  elif name == 'mccnn':
    model = mccnn
  elif name == 'googlenet':
    model = GoogLeNet
    image_size_h_c = 112
    image_size_w_c = 112
    batch_size = 32
  elif name == 'smallvgg':
    model = small_vgg_car

  input2 = (image_size_h_c,image_size_w_c,nchannels)
  if type1 == 'train':
    for k in range(len(keys)):
      K.clear_session()
      val = data[keys[k]]
      aux = keys[:]
      aux.pop(k)
      trn = data[aux[0]] + data[aux[1]]
      tst = data[aux[2]] + data[aux[3]]

      train_steps_per_epoch = ceil(len(trn) / batch_size)
      val_steps_per_epoch = ceil(len(val) / batch_size)
      tst_steps_per_epoch = ceil(len(tst) / batch_size)

      ex1 = ProcessPoolExecutor(max_workers = 4)
      ex2 = ProcessPoolExecutor(max_workers = 4)
      ex3 = ProcessPoolExecutor(max_workers = 4)

      trnGen = generator(trn, batch_size, ex1, input2,  augmentation=True)
      tstGen = generator(val, batch_size, ex2, input2)
      siamese_net = siamese_model(model, input2)

      f1 = 'model_shape_%s_%d.h5' % (name,k)

      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen,
                                    validation_steps=val_steps_per_epoch)

      #validate plate model
      tstGen2 = generator(val, batch_size, ex3, input2, with_paths = True)
      test_report('validation_shape_%s_%d' % (name,k),siamese_net, val_steps_per_epoch, tstGen2)
      del tstGen2
      tstGen2 = generator(tst, batch_size, ex3, input2, with_paths = True)
      test_report('test_shape_%s_%d' % (name,k),siamese_net, tst_steps_per_epoch, tstGen2)

      siamese_net.save(f1)
  elif type1 == 'test':
    results = []
    data = json.load(open(argv[2]))
    img1 = (process_load(data['img1'], input2)/255).reshape(1,input2[0], input2[1],input2[2])
    img2 = (process_load(data['img2'], input2)/255).reshape(1,input2[0], input2[1],input2[2])

    X = [img1, img2,img1, img2]

    for f1 in argv[3:]:
      model = load_model(f1)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model: %s" % ("positive" if results[-1]==POS else "negative"))
    print("final result: %s" % ("positive" if Counter(results).most_common(1)[0][0]==POS else "negative"))
