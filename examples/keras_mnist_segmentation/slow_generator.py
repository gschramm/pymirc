import math
import numpy as np
import os
import threading

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

    # dummy calculation to produce cpu load
    for dd in range(10000):
      dummy = np.random.random(100000)**2

    print(f' creating batch {idx}, pid {os.getpid()}, tid {threading.get_ident()}')
    sleep(self.sleep_time)

    return batch_x, batch_y

#-----------------------------------------------------------------------------------

if __name__ == '__main__':
  shape      = (800,32,32,1)
  val_shape  = (80,32,32,1)
  batch_size = 80
  sleep_time = 0.1

  np.random.seed(1)

  x = np.random.random(shape)
  y = (np.random.random(shape) > 0.5).astype(float)

  gen = SLOWSequence(x, y, batch_size, sleep_time = sleep_time)

  model = unet(input_shape = x.shape[1:], nfeat = 32, batch_normalization = True)

  model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
                loss = keras.losses.BinaryCrossentropy())

  # use_multiporcessing in model.fit() only works correctly in tf 2.1
  # in tf 2.0 it is always executed in the main process
  # in tf 2.0, use fit_generator() (which is deprecated)
  history = model.fit(gen,
                      epochs              = 4,
                      shuffle             = False,
                      use_multiprocessing = True,
                      workers             = 8,
                      max_queue_size      = 8)
