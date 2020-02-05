import math
import numpy as np
from tensorflow    import keras
from scipy.ndimage import rotate

class MNISTSequence(keras.utils.Sequence):

  def __init__(self, x_set, y_set, batch_size, rotation_range_deg = 0):
    self.x, self.y          = x_set, y_set
    self.batch_size         = batch_size
    self.rotation_range_deg = rotation_range_deg

  def __len__(self):
    return math.ceil(self.x.shape[0] / self.batch_size)

  def __getitem__(self, idx):
    start = idx*self.batch_size
    end   = min(self.x.shape[0], (idx + 1)*self.batch_size)

    batch_x = self.x[start:end, ...]
    batch_y = self.y[start:end, ...]

    # data augmentation tricks could be added here
    # here we only apply a random rotation
    if self.rotation_range_deg > 0:
      for i in range(batch_x.shape[0]):
        rot_angle_deg = self.rotation_range_deg * np.random.random_sample(1)[0]
        batch_x[i,...,0] = rotate(batch_x[i,...,0], rot_angle_deg, reshape = False, 
                                  order = 1, prefilter= False)
        batch_y[i,...,0] = rotate(batch_y[i,...,0], rot_angle_deg, reshape = False, 
                                  order = 1, prefilter= False)


    return batch_x, batch_y

  def on_epoch_end(self):
    # !!! this is not working in TF 2.0 with model.fit
    # -> confirmed bug in TF 2.0

    # a work around is to cread a custom callback that
    # takes the generator as input argument and to
    # call generator.on_epoch_end() in the callbacks'
    # on epoch end 

    print('ZZZZZZZ epoch end')
 


