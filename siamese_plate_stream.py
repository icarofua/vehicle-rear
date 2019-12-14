from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config import *
from math import ceil
import json
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Model, load_model
import string
import pandas as pd
from custom_layers import *
from collections import Counter
from sys import argv
import os

#------------------------------------------------------------------------------
def siamese_model(input1):
  left_input_P = Input(input1)
  right_input_P = Input(input1)
  convnet_plate = small_vgg_plate(input1)
  encoded_l_P = convnet_plate(left_input_P)
  encoded_r_P = convnet_plate(right_input_P)
  inputs = [left_input_P, right_input_P]

  # Add the distance function to the network
  L1_distanceP = L1_layer([encoded_l_P, encoded_r_P])
  x = Dense(512, activation='relu')(L1_distanceP)
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
  data = json.load(open('%s/dataset_1.json' % (path)))

  keys = ['Set01','Set02','Set03','Set04','Set05']

  input1 = (image_size_h_p,image_size_w_p,nchannels)
  input2 = (image_size_h_c,image_size_w_c,nchannels)

  type1 = argv[1]

  if type1=='train':

    for k in range(len(keys)):
      K.clear_session()
      val = data[keys[k]]
      aux = keys[:]
      aux.pop(k)
      trn = data[aux[0]] + data[aux[1]]

      train_steps_per_epoch = ceil(len(trn) / batch_size)
      val_steps_per_epoch = ceil(len(val) / batch_size)

      ex1 = ProcessPoolExecutor(max_workers = 4)
      ex2 = ProcessPoolExecutor(max_workers = 4)

      trnGen = generator(trn, batch_size, ex1, input1, input2,  augmentation=True, type='plate')
      tstGen = generator(val, batch_size, ex2, input1, input2, type='plate')
      siamese_net = siamese_model(input1)
      print(siamese_net.summary())

      f1 = 'model_plate_%d.h5' % (k)

      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen,
                                    validation_steps=val_steps_per_epoch)

      #validate plate model
      tstGen2 = generator(val, batch_size, ex2, input1, input2, with_paths = True, type='plate')
      test_report('validation_plate_%d' % (k),siamese_net, val_steps_per_epoch, tstGen2)
      siamese_net.save(f1)
  elif type1 == 'test':
    folder = argv[2]
    for k in range(len(keys)):
      K.clear_session()
      aux = keys[:]
      aux.pop(k)
      tst = data[aux[2]] + data[aux[3]]
      ex3 = ProcessPoolExecutor(max_workers = 4)
      tst_steps_per_epoch = ceil(len(tst) / batch_size)
      tstGen2 = generator(tst, batch_size, ex3, input1, input2, with_paths = True, type='plate')
      f1 = os.path.join(folder,'model_plate_%d.h5' % (k))
      siamese_net = load_model(f1)
      test_report('test_plate_%d' % (k),siamese_net, tst_steps_per_epoch, tstGen2)
  elif type1 == 'predict':
    results = []
    data = json.load(open(argv[2]))
    img1 = (process_load(data['img1'], input1)/255.0).reshape(1,input1[0], input1[1],input1[2])
    img2 = (process_load(data['img2'], input1)/255.0).reshape(1,input1[0], input1[1],input1[2])

    X = [img1, img2]
    folder = argv[3]
    for k in range(len(keys)):
      K.clear_session()
      f1 = os.path.join(folder,'model_plate_%d.h5' % (k))
      model = load_model(f1)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model %d: %s" % (k+1,"positive" if results[k]==POS else "negative"))
    print("final result: %s" % ("positive" if Counter(results).most_common(1)[0][0]==POS else "negative"))
