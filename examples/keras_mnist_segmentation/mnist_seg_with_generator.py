#---------------------------------------------------------------------------------------------
#--- parse command line for input arguments --------------------------------------------------
#---------------------------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description='MNIST segmentation training')
parser.add_argument('--nfeat',         default = 8,      type = int)
parser.add_argument('--batch_size',    default = 50,     type = int)
parser.add_argument('--epochs',        default = 5,      type = int)
parser.add_argument('--oname',         default = None,   type = str )
parser.add_argument('--loss_fct',      default = 'dice', type = str, choices = ('dice','bce'))
parser.add_argument('--no_batch_norm', action = 'store_false', dest = 'batch_norm')

args = parser.parse_args()

import sys, os
sys.path.append(os.path.abspath('../..'))
from tensorflow import keras
from unet       import unet
from mnist_generator import  MNISTSequence
from callbacks import MNISTCallback
from create_data import create_mnist_seg_data
from evaluate_mnist_model import evaluate_mnist_model
from pymirc.metrics.tf_losses import dice

import os
import h5py
import numpy as np
from datetime import datetime

import matplotlib as mpl
try:
  mpl.use('Qt5Agg')
except:
  mpl.use('Agg')
import matplotlib.pyplot as py


#--------------------------------------------------------------------------------------------
#--- initialize parameters ------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

batch_size    = args.batch_size
epochs        = args.epochs
oname         = args.oname
bn            = args.batch_norm
nfeat         = args.nfeat

nval          = 10000
learning_rate = 1e-3
lr_reduce_fac = 0.75
lr_patience   = 10
min_lr        = 1e-4
loss_fct      = args.loss_fct

rot_range_deg = 0

if oname is None:
  dt_str = datetime.now().strftime("%Y%d%m_%H%M%S")
  oname  = f"{dt_str}_ne_{epochs}_bs_{batch_size}_nfeat_{nfeat}_bn_{bn}_loss_{loss_fct}.h5"
  oname  = os.path.join('trained_models', oname)
  os.makedirs('trained_models', exist_ok = True)

np.random.seed(1)

#---------------------------------------------------------------------------------------------
#--- load and normalize the data -------------------------------------------------------------
#---------------------------------------------------------------------------------------------

# create the data if it does not exist
train_fname  = os.path.join('data','mnist_train_seg_data.h5')
if not os.path.exists(train_fname):
  create_mnist_seg_data(train_file = train_fname)

# read the data from disk
with h5py.File(train_fname,'r') as h5data:
  x_train      = h5data['x'][:]
  y_train      = h5data['y'][:]
  train_labels = h5data['label'][:]

# normalize the images such that background is approx 0 and signal approx 1
for i in range(x_train.shape[0]):
  ibg             = np.percentile(x_train[i,...],0.05)
  x_train[i,...] -= ibg
  isignal = -np.percentile(-x_train[i,...],0.05)
  x_train[i,...] /= isignal

#---------------------------------------------------------
# Reserve samples for validation
x_val = x_train[-nval:].astype(np.float32)
y_val = y_train[-nval:].astype(np.float32)
x_train = x_train[:-nval].astype(np.float32)
y_train = y_train[:-nval].astype(np.float32)
val_labels   = train_labels[-nval:]
train_labels = train_labels[:-nval]

input_shape = x_train.shape[1:]

#---------------------------------------
# set up the data generator for training
mnist_train_gen  =  MNISTSequence(x_train, y_train, batch_size, rotation_range_deg = rot_range_deg)


#---------------------------------------------------------------------------------------------
#--- setup and train the model ---------------------------------------------------------------
#---------------------------------------------------------------------------------------------

if loss_fct == 'dice':
  loss = dice
elif loss_fct == 'bce':
  loss = keras.losses.BinaryCrossentropy()

model = unet(input_shape = input_shape, nfeat = nfeat, batch_normalization = bn)

model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
              loss      = loss)

#-----------------------------------------------------------------
# train the model

# define a callback that reduces the learning rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor  = 'val_loss',
                                              factor   = lr_reduce_fac,
                                              patience = lr_patience,
                                              min_lr   = min_lr)

mc = keras.callbacks.ModelCheckpoint(oname, monitor='val_loss', mode='min',
                                     save_best_only=True, verbose = 1)

history = model.fit(mnist_train_gen,
                    epochs              = epochs,
                    validation_data     = (x_val, y_val),
                    callbacks           = [reduce_lr, MNISTCallback(), mc],
                    shuffle             = False)

# The returned "history" object holds a record
# of the loss values and metric values during training

#------------------------------------
# plot the loss functions

py.rcParams['axes.titlesize'] = 'medium'
fig3, ax3 = py.subplots(figsize = (4,4))
ax3.plot(history.history['loss'], label = 'train')
ax3.plot(history.history['val_loss'], label = 'val')
ax3.legend()
ax3.set_title('loss')
ax3.grid(ls = ':')
fig3.tight_layout()
fig3.savefig(oname.replace('.h5','_fig3.png'))
fig3.show()

#---------------------------------------------------------------------------------------------
#--- evaluated the model and show some results -----------------------------------------------
#---------------------------------------------------------------------------------------------

evaluate_mnist_model(oname, x_val, y_val, val_labels)
