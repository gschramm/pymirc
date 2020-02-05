from tensorflow import keras
from scipy.ndimage import gaussian_filter, zoom
import numpy as np
import h5py
import os

def create_mnist_seg_data(train_file      = os.path.join('data','mnist_train_seg_data.h5'),
                          test_file       = os.path.join('data','mnist_test_seg_data.h5'),
                          seed            = 42,
                          cnr_range       = [1.2,1.5],
                          bg_range        = [0,2],
                          ps_sigma_range  = [1,1.5],
                          zoom_range      = [1,1.7],
                          max_shift       = 23):
  np.random.seed(42)

  # test if outpur dir exists and create it if necessary
  os.makedirs(os.path.dirname(train_file), exist_ok=True)
 
  # Load a toy dataset for the sake of this example
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  
  label_train = y_train.copy()
  label_test  = y_test.copy()
  
  # Preprocess the data (these are Numpy arrays)
  seg_train = np.expand_dims(x_train,-1).astype('float32')
  seg_test  = np.expand_dims(x_test,-1).astype('float32')
  
  seg_train /= seg_train.max()
  seg_test  /= seg_test.max()
  
  seg_train = (seg_train > 0.5).astype(np.float)
  seg_test  = (seg_test > 0.5).astype(np.float)

  imshape = (64,64)  

  # prepare the training data
  x_train = np.zeros((seg_train.shape[0],) + imshape + (1,))
  y_train = np.zeros((seg_train.shape[0],) + imshape + (1,))

  for i in range(seg_train.shape[0]):
    cnr      = (cnr_range[1] - cnr_range[0])*np.random.random_sample() + cnr_range[0]
    bg       = (bg_range[1] - bg_range[0])*np.random.random_sample() + bg_range[0]
    ps_sigma = (ps_sigma_range[1] - ps_sigma_range[0])*np.random.random_sample() + ps_sigma_range[0]
    sign     = 2*int(np.random.random_sample() > 0.5) - 1
    zoom_fac = (zoom_range[1] - zoom_range[0])*np.random.random_sample() + zoom_range[0]
  
    shift0   = np.random.randint(0,max_shift) - max_shift // 2
    shift1   = np.random.randint(0,max_shift) - max_shift // 2
  
    seg    = zoom(seg_train[i,...,0], zoom = zoom_fac, order = 0, prefilter = False)
    pad0   = (imshape[0] - seg.shape[0]) // 2
    pad1   = (imshape[0] - seg.shape[0]) - pad0
  
    seg    = np.pad(seg, (pad0,pad1))
    seg    = np.roll(seg, (shift0, shift1), axis = (0,1))
  
    tmp = cnr*seg + np.random.randn(*imshape)
    x_train[i,...,0] = sign*(gaussian_filter(tmp, sigma = ps_sigma) + bg)
    y_train[i,...,0] = seg

  # prepare the test data
  x_test = np.zeros((seg_test.shape[0],) + imshape + (1,))
  y_test = np.zeros((seg_test.shape[0],) + imshape + (1,))

  for i in range(seg_test.shape[0]):
    cnr      = (cnr_range[1] - cnr_range[0])*np.random.random_sample() + cnr_range[0]
    bg       = (bg_range[1] - bg_range[0])*np.random.random_sample() + bg_range[0]
    ps_sigma = (ps_sigma_range[1] - ps_sigma_range[0])*np.random.random_sample() + ps_sigma_range[0]
    sign     = 2*int(np.random.random_sample() > 0.5) - 1
    zoom_fac = (zoom_range[1] - zoom_range[0])*np.random.random_sample() + zoom_range[0]
  
    shift0   = np.random.randint(0,max_shift) - max_shift // 2
    shift1   = np.random.randint(0,max_shift) - max_shift // 2
  
    seg    = zoom(seg_test[i,...,0], zoom = zoom_fac, order = 0, prefilter = False)
    pad0   = (imshape[0] - seg.shape[0]) // 2
    pad1   = (imshape[0] - seg.shape[0]) - pad0
  
    seg    = np.pad(seg, (pad0,pad1))
    seg    = np.roll(seg, (shift0, shift1))
  
    tmp = cnr*seg + np.random.randn(*imshape)
    x_test[i,...,0] = sign*(gaussian_filter(tmp, sigma = ps_sigma) + bg)
    y_test[i,...,0] = seg

  # write the data to hdf5
  with h5py.File(train_file, 'w') as h5f:
     h5f.create_dataset('x',     data = x_train)
     h5f.create_dataset('y',     data = y_train)
     h5f.create_dataset('label', data = label_train)
  print('wrote', train_file)

  with h5py.File(test_file, 'w') as h5f:
     h5f.create_dataset('x',     data = x_test)
     h5f.create_dataset('y',     data = y_test)
     h5f.create_dataset('label', data = label_test)
  print('wrote', test_file)
  
