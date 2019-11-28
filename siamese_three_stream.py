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

metadata_dict = {}

#------------------------------------------------------------------------------
def read_metadata(labels):
  global metadata_dict
  data = pd.read_csv(ocr_file, sep=' ')
  ocr_dict = {}
  #"{0:05b}".format(10)
  alpha_dict = {i.upper():j/35 for j,i in enumerate(string.ascii_uppercase + string.digits)}
  data.fillna(0, inplace=True)
  for i in data.index:
    key = "/".join(data.loc[i,"file"].split("/")[-5:])
    ocrs = []

    for char1 in data.loc[i,'pred']:
      ocrs.append(alpha_dict[char1])

    if len(ocrs)<7:
      ocrs+=[0]*(7-len(ocrs))

    for j in range(1,8):
      ocrs.append(data.loc[i,'char%d' % j])

    ocr_dict[key] = ocrs

  for i in labels:
    key = "/".join(i.split("/")[-5:])
    if key in ocr_dict:
      metadata_dict[i] = ocr_dict[key]
    else:
      metadata_dict[i] = [0] * 14

  del ocr_dict, data, alpha_dict
  return metadata_dict
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
  auxiliary_input = Input(shape=(metadata_length,), name='aux_input')
  inputs = [left_input_P, right_input_P, left_input_C, right_input_C, auxiliary_input]

  # Add the distance function to the network
  L1_distanceP = L1_layer([encoded_l_P, encoded_r_P])
  L1_distanceC = L1_layer([encoded_l_C, encoded_r_C])
  x = Concatenate()([L1_distanceP, L1_distanceC, auxiliary_input])
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(1024, kernel_initializer='normal',activation='relu')(x)
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

  labels = []
  for k in keys:
    for img in data[k]:
      labels += [img[0][0], img[2][0]]
  labels = list(set(labels))
  read_metadata(labels)

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

      f1 = 'model_three_stream_%d.h5' % (k)

      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen,
                                    validation_steps=val_steps_per_epoch)

      #validate plate model
      tstGen2 = generator(val, batch_size, ex3, input1, input2,  with_paths = True)
      test_report('validation_three_stream_%d' % (k),siamese_net, val_steps_per_epoch, tstGen2)
      del tstGen2
      tstGen2 = generator(tst, batch_size, ex3, input1, input2,  with_paths = True)
      test_report('test_three_stream_%d' % (k),siamese_net, tst_steps_per_epoch, tstGen2)

      siamese_net.save(f1)
  elif type1 == 'test':
    results = []
    data = json.load(open(argv[2]))
    alpha_dict = {i.upper():j/35 for j,i in enumerate(string.ascii_uppercase + string.digits)}

    img1 = (process_load(data['img1_plate'], input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img2 = (process_load(data['img2_plate'], input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img3 = (process_load(data['img1_shape'], input2)/255.0).reshape(1,input2[0],input2[1],input2[2])
    img4 = (process_load(data['img2_shape'], input2)/255.0).reshape(1,input2[0],input2[1],input2[2])

    aux1 = []
    for str1 in data['ocr1']:
      for c in str1:
        aux1.append(alpha_dict[c])
    aux1 += data['probs1']

    aux2 = []
    for str1 in data['ocr2']:
      for c in str1:
        aux2.append(alpha_dict[c])
    aux2 += data['probs2']

    diff = abs(np.array(aux1[:7]) - np.array(aux2[:7])).tolist()
    for j in range(len(diff)):
      diff[j] = 1 if diff[j] else 0
    metadata = aux1 + aux2 + diff

    metadata = np.array(metadata).reshape(1,-1)

    X = [img1, img2, img3, img4, metadata]

    k = 0
    for f1 in argv[3:]:
      model = load_model(f1)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model %d: %s" % (k+1,"positive" if results[k]==POS else "negative"))
      k+=1
    print("final result: %s" % ("positive" if Counter(results).most_common(1)[0][0]==POS else "negative"))
