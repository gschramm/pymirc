# taken from https://github.com/zhixuhao/unet/blob/master/model.py
from tensorflow import keras

def unet(input_shape = (32,32,1), nfeat = 8, batch_normalization = False):
  inputs = keras.layers.Input(input_shape)
  conv1  = keras.layers.Conv2D(nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1  = keras.layers.Conv2D(nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  if batch_normalization:
    conv1 = keras.layers.BatchNormalization()(conv1)
  pool1  = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2  = keras.layers.Conv2D(2*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2  = keras.layers.Conv2D(2*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  if batch_normalization:
    conv2 = keras.layers.BatchNormalization()(conv2)
  pool2  = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3  = keras.layers.Conv2D(4*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3  = keras.layers.Conv2D(4*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  if batch_normalization:
    conv3 = keras.layers.BatchNormalization()(conv3)
  pool3  = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv4  = keras.layers.Conv2D(8*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4  = keras.layers.Conv2D(8*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  if batch_normalization:
    conv4 = keras.layers.BatchNormalization()(conv4)
  drop4  = keras.layers.Dropout(0.5)(conv4)
  pool4  = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5  = keras.layers.Conv2D(16*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5  = keras.layers.Conv2D(16*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  if batch_normalization:
    conv5 = keras.layers.BatchNormalization()(conv5)
  drop5  = keras.layers.Dropout(0.5)(conv5)

  up6    = keras.layers.Conv2D(8*nfeat, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5))
  merge6 = keras.layers.concatenate([drop4,up6], axis = 3)
  conv6  = keras.layers.Conv2D(8*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6  = keras.layers.Conv2D(8*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
  if batch_normalization:
    conv6 = keras.layers.BatchNormalization()(conv6)

  up7    = keras.layers.Conv2D(4*nfeat, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6))
  merge7 = keras.layers.concatenate([conv3,up7], axis = 3)
  conv7  = keras.layers.Conv2D(4*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7  = keras.layers.Conv2D(4*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
  if batch_normalization:
    conv7 = keras.layers.BatchNormalization()(conv7)

  up8    = keras.layers.Conv2D(2*nfeat, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))
  merge8 = keras.layers.concatenate([conv2,up8], axis = 3)
  conv8  = keras.layers.Conv2D(2*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8  = keras.layers.Conv2D(2*nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
  if batch_normalization:
    conv8 = keras.layers.BatchNormalization()(conv8)

  up9    = keras.layers.Conv2D(nfeat, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))
  merge9 = keras.layers.concatenate([conv1,up9], axis = 3)
  conv9  = keras.layers.Conv2D(nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9  = keras.layers.Conv2D(nfeat, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9  = keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  if batch_normalization:
    conv9 = keras.layers.BatchNormalization()(conv9)

  conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

  model  = keras.Model(inputs, conv10)

  return model
