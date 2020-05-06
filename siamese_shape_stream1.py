from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from config_1 import *
import json
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Model, load_model
from sys import argv
from custom_layers import *
from collections import Counter
import os
#------------------------------------------------------------------------------
def siamese_model(model, input2):
  left_input_C = Input(input2)
  right_input_C = Input(input2)
  convnet_car = model(input2)
  encoded_l_C = convnet_car(left_input_C)
  encoded_r_C = convnet_car(right_input_C)
  inputs = [left_input_C, right_input_C]

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
  model.compile(loss=losses, loss_weights=lossWeights,optimizer=optimizer, metrics=kmetrics)

  return model
#------------------------------------------------------------------------------
if __name__ == '__main__':

  data = json.load(open('%s/dataset_1.json' % (path)))

  type1 = argv[1]
  name = argv[2]
  if name == 'resnet50':
    model = resnet50_model
    image_size_h_c = 224
    image_size_w_c = 224
    batch_size = 8
  elif name == 'vgg16':
    model = vgg16_model
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
    image_size_h_c = 128
    image_size_w_c = 128
  elif name == 'matchnet':
    model = matchnet
    image_size_h_c = 128
    image_size_w_c = 128
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
    for k,val_idx in enumerate(keys):
      K.clear_session()
      idx = fold(keys,k, train=True)
      val = data[val_idx]
      trn = data[idx[0]] + data[idx[1]]

      trnGen = SiameseSequence(trn, train_augs,batch_size=batch_size,input2=input2, type1='car')
      tstGen = SiameseSequence(val, test_augs,batch_size=batch_size,input2=input2, type1='car')
      siamese_net = siamese_model(model, input2)

      f1 = 'model_shape_%s_%d.h5' % (name,k)

      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen)
      #validate plate model
      tstGen2 = SiameseSequence(val, test_augs, batch_size=batch_size,input2=input2, with_paths=True, type1='car')
      test_report('validation_shape_%s_%d' % (name,k),siamese_net, tstGen2)

      siamese_net.save(f1)
  elif type1 == 'test':
    folder = argv[3]
    for k in range(len(keys)):
      idx = fold(keys,k, train=False)
      tst = data[idx[0]] + data[idx[1]]
      tstGen2 = SiameseSequence(tst, test_augs,batch_size=batch_size,input2=input2, with_paths=True, type1='car')
      f1 = os.path.join(folder,'model_shape_%s_%d.h5' % (name, k))
      siamese_net = load_model(f1, custom_objects=customs_func)
      test_report('test_shape_%s_%d' % (name, k),siamese_net, tstGen2)
  elif type1 == 'predict':

    results = []
    data = json.load(open(argv[3]))
    img3 = (process_load(data['img1'], input2)/255).reshape(1,input2[0], input2[1],input2[2])
    img4 = (process_load(data['img2'], input2)/255).reshape(1,input2[0], input2[1],input2[2])

    X = [img3, img4]

    folder = argv[4]
    for k in range(len(keys)):
      K.clear_session()
      f1 = os.path.join(folder,'model_shape_%s_%d.h5' % (name, k))
      model = load_model(f1, custom_objects=customs_func)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model %d: %s" % (k+1,"positive" if results[k]==POS else "negative"))
    print("final result: %s" % ("positive" if Counter(results).most_common(1)[0][0]==POS else "negative"))
