import numpy as np
import tensorflow as tf
if tf.__version__.startswith('1.'):
  from keras import backend as K
else:
  from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):
    '''Returns the dice coefficient after thresholding the predictions at
    a 0.5 confidence level.

    dice = 2*intersection / union

    '''
    threshold = 0.5
    y_true = K.cast(K.greater(y_true, threshold), 'float32')
    y_pred = K.cast(K.greater(y_pred, threshold), 'float32')
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    # avoid division by zero by adding 1
    dice = (2.0 * intersection + K.epsilon()) / (union + K.epsilon())
    return dice


def soft_dice_coef(y_true, y_pred):
    '''Returns the soft dice coefficient.

    This is calculated the same way as the normal dice coefficient, but
    without thresholding. This provides a continuous metric but with the same
    trend as the actual dice coefficient. This is especially useful as a loss
    function.

    '''
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    dice = (2.0 * intersection + K.epsilon()) / (union + K.epsilon())
    return dice


def jaccard_index(y_true, y_pred):
    '''Returns the jaccard index

    jaccard index = intersection / union
    '''
    threshold = 0.5
    y_true = K.cast(K.greater(y_true, threshold), 'float32')
    y_pred = K.cast(K.greater(y_pred, threshold), 'float32')
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    jaccard = (intersection + K.epsilon()) / (union - intersection + K.epsilon())  # this contains as many elements as there are classes
    return jaccard


def soft_jaccard_index(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    jaccard = (intersection + K.epsilon()) / (union - intersection + K.epsilon())
    return jaccard


def IoU(y_true, y_pred):
    '''Returns the intersection over union


    This is mathematically equivalent to the jaccard index.
    '''
    return jaccard_index(y_true, y_pred)

def tf_gauss_kernel_3d(sigma, size):
  """Generate 3D Gaussian kernel 
  
  Parameters
  ----------

  sigma : float
    width of the gaussian
  size : int 
    size of the gaussian (should be odd an approx 2*int(3.5*sigma + 0.5) + 1

  Returns
  -------
  tensorflow tensor with dimension [size,size,size,1,1] with tf.reduce_sum(k) = 1
  """
  size  = tf.convert_to_tensor(size, tf.int32)
  sigma = tf.convert_to_tensor(sigma, tf.float32)

  coords = tf.cast(tf.range(size), tf.float32) - tf.cast(size - 1, tf.float32) / 2.0

  g = -0.5*tf.square(coords) / tf.square(sigma)
  g = tf.nn.softmax(g)

  g = tf.einsum('i,j,k->ijk', g, g, g)
  g = tf.expand_dims(tf.expand_dims(g, -1), -1)

  return g

def ssim_3d(x, y, sigma = 1.5, size = 11, L = None, K1 = 0.01, K2  = 0.03, return_image = False):
  """ Compute the structural similarity between two batches of 3D single channel images

  Parameters
  ----------

  x,y : tensorflow tensors with shape [batch_size,depth,height,width,1] 
    containing a batch of 3D images with 1 channel
  L : float
    dynamic range of the images. 
    By default (None) it is set to tf.reduce_max(y) - tf.reduce_min(y)
  K1, K2 : float
    small constants needed to avoid division by 0 see [1]. 
    Default 0.01, 0.03
  sigma : float 
    width of the gaussian filter in pixels
    Default 1.5
  size : int
    size of the gaussian kernel used to calculate local means and std.devs 
    Default 11

  Returns
  -------
  a 1D tensorflow tensor of length batch_size containing the SSIM for
  every image pair in the batch

  Note
  ----
  (1) This implementation is very close to [1] and 
      from skimage.metrics import structural_similarity
      structural_similarity(x, y, gaussian_weights = True, full = True, data_range = L)
  (2) The default way of how the dynamic range L is calculated (based on y)
      is different from [1] and structural_similarity()

  References
  ----------
  [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
  (2004). Image quality assessment: From error visibility to
  structural similarity. IEEE Transactions on Image Processing
  """
  if (x.shape[-1] != 1) or (y.shape[-1] != 1):
    raise ValueError('Last dimension of input x has to be 1')

  if L is None:
      L = tf.reduce_max(y) - tf.reduce_min(y)
    
  C1 = (K1*L)**2
  C2 = (K2*L)**2
  
  shape  = x.shape
  kernel = tf_gauss_kernel_3d(sigma, size)

  mu_x     = tf.nn.conv3d(x, kernel, strides = [1,1,1,1,1], padding = 'VALID')
  mu_y     = tf.nn.conv3d(y, kernel, strides = [1,1,1,1,1], padding = 'VALID')

  mu_x_sq  = mu_x*mu_x
  mu_y_sq  = mu_y*mu_y
  mu_x_y   = mu_x*mu_y

  sig_x_sq = tf.nn.conv3d(x*x, kernel, strides = [1,1,1,1,1], padding = 'VALID') - mu_x_sq
  sig_y_sq = tf.nn.conv3d(y*y, kernel, strides = [1,1,1,1,1], padding = 'VALID') - mu_y_sq
  sig_xy   = tf.nn.conv3d(x*y, kernel, strides = [1,1,1,1,1], padding = 'VALID') - mu_x_y
  
  SSIM= (2*mu_x_y + C1)*(2*sig_xy + C2) / ((mu_x_sq + mu_y_sq + C1)*(sig_x_sq + sig_y_sq + C2))

  if not return_image:
    SSIM = tf.reduce_mean(SSIM, [1,2,3,4]) 

  return SSIM

# aliases
# dice = DICE = dice_coef
# soft_dice = soft_dice_coef
# iou = IOU = IoU
# jaccard = jaccard_index
