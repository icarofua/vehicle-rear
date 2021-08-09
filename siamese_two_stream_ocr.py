from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from config import *
import json
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Model, load_model
from sys import argv
from custom_layers import *
from collections import Counter
import os
import string
import pandas as pd

def read_metadata(labels):
  metadata_dict = {}

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
def siamese_model(input2):
  left_input_C = Input(input2)
  right_input_C = Input(input2)
  auxiliary_input = Input(shape=(metadata_length,), name='aux_input')
  convnet_car = small_vgg_car(input2)
  encoded_l_C = convnet_car(left_input_C)
  encoded_r_C = convnet_car(right_input_C)
  inputs = [left_input_C, right_input_C, auxiliary_input]

  # Add the distance function to the network
  x = L1_layer([encoded_l_C, encoded_r_C])
  x = Concatenate()([x, auxiliary_input])
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
  model.compile(loss=losses, loss_weights=lossWeights,optimizer=optimizer,metrics=kmetrics)

  return model
#------------------------------------------------------------------------------
if __name__ == '__main__':
  data = json.load(open('%s/dataset_1.json' % (path)))

  labels = []
  for k in keys:
    for img in data[k]:
      labels += [img[0][0], img[2][0]]
  labels = list(set(labels))
  metadata_dict = read_metadata(labels)

  input1 = (image_size_h_p,image_size_w_p,nchannels)
  input2 = (image_size_h_c,image_size_w_c,nchannels)
  type1 = argv[1]
  if type1=='train':
    for k,val_idx in enumerate(keys):
      K.clear_session()
      idx = fold(keys,k, train=True)
      val = data[val_idx]
      trn = data[idx[0]] + data[idx[1]]

      trnGen = SiameseSequence(trn, train_augs, type1='car', metadata_dict=metadata_dict,metadata_length=metadata_length)
      tstGen = SiameseSequence(val, test_augs, type1='car', metadata_dict=metadata_dict,metadata_length=metadata_length)

      siamese_net = siamese_model(input2)
      print(siamese_net.summary())
      f1 = 'model_two_stream_ocr_%d.h5' % (k)

      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen)

      #validate plate model
      tstGen2 = SiameseSequence(val, test_augs, metadata_dict=metadata_dict,metadata_length=metadata_length, with_paths = True, type1='car')
      test_report('validation_two_stream_ocr_%d' % (k),siamese_net, tstGen2)

    siamese_net.save(f1)

  elif type1 == 'test':
    folder = argv[2]
    for k in range(len(keys)):
      idx = fold(keys,k, train=False)
      tst = data[idx[0]] + data[idx[1]]
      tstGen2 = SiameseSequence(tst, test_augs, metadata_dict=metadata_dict,metadata_length=metadata_length, with_paths = True, type1='car')
      f1 = os.path.join(folder,'model_two_stream_ocr_%d.h5' % (k))
      siamese_net = load_model(f1, custom_objects=customs_func)
      test_report('test_two_stream_ocr_%d' % (k),siamese_net, tstGen2)
  elif type1 == 'predict':
    results = []
    data = json.load(open(argv[2]))
    alpha_dict = {i.upper():j/35 for j,i in enumerate(string.ascii_uppercase + string.digits)}
    img3 = (process_load(data['img1'], input2)/255.0).reshape(1, input2[0], input2[1], input2[2])
    img4 = (process_load(data['img2'], input2)/255.0).reshape(1, input2[0], input2[1], input2[2])

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

    X = [img3, img4, metadata]

    folder = argv[3]
    for k in range(len(keys)):
      K.clear_session()
      f1 = os.path.join(folder,'model_two_stream_ocr_%d.h5' % (k))
      model = load_model(f1, custom_objects=customs_func)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model %d: %s" % (k+1,"positive" if results[k]==POS else "negative"))
    print("final result: %s" % ("positive" if Counter(results).most_common(1)[0][0]==POS else "negative"))
