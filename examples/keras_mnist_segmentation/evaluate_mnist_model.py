from tensorflow import keras
from pymirc.metrics.tf_losses import dice
import numpy as np

import matplotlib as mpl
try:
  mpl.use('Qt5Agg')
except:
  mpl.use('Agg')
import matplotlib.pyplot as py

#-----------------------------------------------------------
def evaluate_mnist_model(fname, x_val, y_val, val_labels): 

  model = keras.models.load_model(fname, custom_objects = {'dice': dice})
  
  # Generate predictions (probabilities -- the output of the last layer)
  # on new data using `predict`
  predictions  = model.predict(x_val)
  predictions2 = model.predict(np.swapaxes(x_val,1,2))
  
  val_dice  = np.zeros(10)
  val_dice2 = np.zeros(10)

  val_dice_std  = np.zeros(10)
  val_dice_std2 = np.zeros(10)
  
  for i in range(10):
    inds = np.where(val_labels == i)
    d1 = dice(predictions[inds],y_val[inds]).numpy()
    d2 = dice(predictions2[inds],np.swapaxes(y_val[inds],1,2)).numpy()
    val_dice[i]       = 1 - d1.mean() 
    val_dice2[i]      = 1 - d2.mean() 
    val_dice_std[i]   = d1.std() 
    val_dice_std2[i]  = d2.std() 

  np.savetxt(fname.replace('.h5','_val_dice.txt'), 
             np.array([val_dice,val_dice_std,val_dice2,val_dice_std2]).transpose(), delimiter = ' ')
 
  print('val dice coefficients')
  for i in range(10): print('label:', i, val_dice[i], val_dice2[i])
  
  # show some results
  
  py.rcParams['axes.titlesize'] = 'x-small'
  py.rcParams['image.cmap']     = 'Greys'
  
  #------------------------------------
  # show the first 100 wrong predictions
  n_show = 100
  ncols  = 30
  nrows  = 10
  
  fig, ax  = py.subplots(nrows, ncols, figsize=(15, 5))
  for i, axx in enumerate(ax.flatten()):
    ii = i // 3
    if i % 3 == 0:
      axx.imshow(x_val[ii,:,:,0], vmin = 0, vmax = 1)
    elif i % 3 == 1:
      axx.imshow(y_val[ii,:,:,0], vmin = 0, vmax = 1)
    else:
      axx.imshow(predictions[ii,:,:,0], vmin = 0, vmax = 1)
    axx.set_axis_off()
  fig.tight_layout()
  fig.savefig(fname.replace('.h5','_fig1.png'))
  fig.show()
  
  fig2, ax2  = py.subplots(nrows, ncols, figsize=(15, 5))
  for i, axx in enumerate(ax2.flatten()):
    ii = i // 3
    if i % 3 == 0:
      axx.imshow(np.swapaxes(x_val[ii,:,:,0],0,1), vmin = 0, vmax = 1)
    elif i % 3 == 1:
      axx.imshow(np.swapaxes(y_val[ii,:,:,0],0,1), vmin = 0, vmax = 1)
    else:
      axx.imshow(predictions2[ii,:,:,0], vmin = 0, vmax = 1)
    axx.set_axis_off()
  fig2.tight_layout()
  fig2.savefig(fname.replace('.h5','_fig2.png'))
  fig2.show()
