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
from sys import argv
from custom_layers import *
from collections import Counter

#------------------------------------------------------------------------------
def siamese_model(input1, input2):
  left_input_P = Input(input1)
  right_input_P = Input(input1)
  left_input_C = Input(input2)
  right_input_C = Input(input2)
  convnet_plate = small_vgg_plate(input1)
  encoded_l_P = convnet_plate(left_input_P)
  encoded_r_P = convnet_plate(right_input_P)
  convnet_car = small_vgg_car(input2)
  encoded_l_C = convnet_car(left_input_C)
  encoded_r_C = convnet_car(right_input_C)
  inputs = [left_input_P, right_input_P, left_input_C, right_input_C]

  # Add the distance function to the network
  L1_distanceP = L1_layer([encoded_l_P, encoded_r_P])
  L1_distanceC = L1_layer([encoded_l_C, encoded_r_C])
  concatL1 = Concatenate()([L1_distanceP, L1_distanceC])
  x = Dense(1024, activation='relu')(concatL1)
  x = Dense(1024, kernel_initializer='normal',activation='relu')(x)
  x = Dense(1024, kernel_initializer='normal',activation='relu')(x)
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
      tst = data[aux[2]] + data[aux[3]]

      train_steps_per_epoch = ceil(len(trn) / batch_size)
      val_steps_per_epoch = ceil(len(val) / batch_size)
      tst_steps_per_epoch = ceil(len(tst) / batch_size)

      ex1 = ProcessPoolExecutor(max_workers = 4)
      ex2 = ProcessPoolExecutor(max_workers = 4)
      ex3 = ProcessPoolExecutor(max_workers = 4)

      trnGen = generator(trn, batch_size, ex1, input1, input2,  augmentation=True)
      tstGen = generator(val, batch_size, ex2, input1, input2)
      siamese_net = siamese_model(input1, input2)

      f1 = 'model_two_stream_%d.h5' % (k)

      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen,
                                    validation_steps=val_steps_per_epoch)

      #validate plate model
      tstGen2 = generator(val, batch_size, ex3, input1, input2, with_paths = True)
      test_report('validation_two_stream_%d' % (k),siamese_net, val_steps_per_epoch, tstGen2)
      del tstGen2
      tstGen2 = generator(tst, batch_size, ex3, input1, input2, with_paths = True)
      test_report('test_two_stream_%d' % (k),siamese_net, tst_steps_per_epoch, tstGen2)

      siamese_net.save(f1)
  elif type1 == 'test':
    results = []
    data = json.load(open(argv[2]))

    img1 = process_load(data['img1_plate'], input1)/255.0
    img2 = process_load(data['img2_plate'], input1)/255.0
    img3 = process_load(data['img1_shape'], input2)/255.0
    img4 = process_load(data['img2_shape'], input2)/255.0

    X = [img1, img2, img3, img4]

    for f1 in argv[3:]:
      model = load_model(f1)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model %d: %s",k+1,"positive" if results[k]==POS else "negative")
    print("final result: %s","positive" if Counter(results).most_common(1)==POS else "negative")
