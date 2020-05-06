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
  model.compile(loss=losses, loss_weights=lossWeights,optimizer=optimizer, metrics=kmetrics)
  return model
#------------------------------------------------------------------------------
if __name__ == '__main__':
  data = json.load(open('%s/dataset_1.json' % (path)))

  input1 = (image_size_h_p,image_size_w_p,nchannels)
  input2 = (image_size_h_c,image_size_w_c,nchannels)
  type1 = argv[1]
  if type1=='train':
    for k,val_idx in enumerate(keys):
      K.clear_session()
      idx = fold(keys,k, train=True)
      val = data[val_idx]
      trn = data[idx[0]] + data[idx[1]]

      trnGen = SiameseSequence(trn, train_augs)
      tstGen = SiameseSequence(val, test_augs)
      siamese_net = siamese_model(input1, input2)

      f1 = 'model_two_stream_%d.h5' % (k)

      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen)

      #validate plate model
      tstGen2 = SiameseSequence(val, test_augs, with_paths=True)
      test_report('validation_two_stream_%d' % (k),siamese_net, tstGen2)

      siamese_net.save(f1)

  elif type1 == 'test':
    folder = argv[2]
    for k in range(len(keys)):
      idx = fold(keys,k, train=False)
      tst = data[idx[0]] + data[idx[1]]
      tstGen2 = SiameseSequence(tst, test_augs, with_paths=True)
      f1 = os.path.join(folder,'model_two_stream_%d.h5' % (k))
      siamese_net = load_model(f1, custom_objects=customs_func)
      test_report('test_two_stream_%d' % (k),siamese_net, tstGen2)
  elif type1 == 'predict':

    results = []
    data = json.load(open(argv[2]))

    img1 = (process_load(data['img1_plate'], input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img2 = (process_load(data['img2_plate'], input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img3 = (process_load(data['img1_shape'], input2)/255.0).reshape(1,input2[0],input2[1],input2[2])
    img4 = (process_load(data['img2_shape'], input2)/255.0).reshape(1,input2[0],input2[1],input2[2])

    X = [img1, img2, img3, img4]

    folder = argv[3]
    for k in range(len(keys)):
      K.clear_session()
      f1 = os.path.join(folder,'model_two_stream_%d.h5' % (k))
      model = load_model(f1)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model %d: %s" % (k+1,"positive" if results[k]==POS else "negative"))
    print("final result: %s" % ("positive" if Counter(results).most_common(1)[0][0]==POS else "negative"))

