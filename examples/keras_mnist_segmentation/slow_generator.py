import math
import numpy as np

from tensorflow import keras
from unet import unet
from time import sleep

class SLOWSequence(keras.utils.Sequence):

  def __init__(self, x_set, y_set, batch_size, sleep_time = 0.1):
    self.x, self.y          = x_set, y_set
    self.batch_size         = batch_size
    self.sleep_time         = sleep_time

  def __len__(self):
    return math.ceil(self.x.shape[0] / self.batch_size)

  def __getitem__(self, idx):
    start = idx*self.batch_size
    end   = min(self.x.shape[0], (idx + 1)*self.batch_size)

    batch_x = self.x[start:end, ...]
    batch_y = self.y[start:end, ...]

    print(f'going to sleep for {sleep_time}s')
    sleep(self.sleep_time)

    return batch_x, batch_y

#-----------------------------------------------------------------------------------

if __name__ == '__main__':
  shape      = (800,32,32,1)
  val_shape  = (80,32,32,1)
  batch_size = 8
  sleep_time = 1

  np.random.seed(1)

  x = np.random.random(shape)
  y = (np.random.random(shape) > 0.5).astype(float)

  x_val = np.random.random(val_shape)
  y_val = (np.random.random(val_shape) > 0.5).astype(float)

  gen = SLOWSequence(x, y, batch_size, sleep_time = sleep_time)

  model = unet(input_shape = x.shape[1:], nfeat = 2, batch_normalization = True)

  model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
                loss = keras.losses.BinaryCrossentropy())

  history = model.fit(gen,
                      epochs              = 2,
                      validation_data     = (x_val, y_val),
                      shuffle             = False)
