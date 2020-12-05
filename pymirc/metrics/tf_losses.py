import tensorflow as tf
if tf.__version__.startswith('1.'):
  from keras import backend as K
else:
  from tensorflow.keras import backend as K

from .tf_metrics import soft_dice_coef, soft_dice_coef_3d, soft_jaccard_index, IoU, ssim_3d, generalized_dice_coeff


def weighted_binary_crossentropy(weights=[.5, 1]):
    '''Returns a weighted binary crossentropy function.

    Arguments:
        weights - First element of list is weight given to the background
            and the second element to the foreground (i.e. where the GT is
            equal to 1)
    '''
    def weighted_binary_crossentropy(y_true, y_pred):
        weight_mask = (1 - y_true) * weights[0] + y_true * weights[1]
        return K.mean(
            weight_mask * K.binary_crossentropy(y_pred, y_true), axis=-1)

    return weighted_binary_crossentropy


def dice(y_true, y_pred):
    '''
    Equal to 1 minus the soft dice coefficient defined in metrics
    '''
    return 1 - soft_dice_coef(y_true, y_pred)


def dice_3d(y_true, y_pred):
    '''
    Equal to 1 minus the soft dice coefficient defined in metrics
    '''
    return 1 - soft_dice_coef_3d(y_true, y_pred)


def binary_crossentropy(y_true, y_pred):
    '''Redefinition of keras's binary crossentropy'''
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def jaccard(y_true, y_pred):
    '''Equal to 1 minus the jaccard index defined in metrics'''
    return 1 - soft_jaccard_index(y_true, y_pred)


def IoU_loss(y_true, y_pred):
    '''Equal to 1 minus the intersection over union defined in metrics'''
    return 1 - IoU(y_true, y_pred)


def focal_loss(gamma=2., alpha=.25, from_logits=False):
    '''Focal loss

    Focal loss as defined in:
    Lin, Tsung-Yi, et al. "Focal loss for dense object detection." arXiv preprint arXiv:1708.02002 (2017).

    This is a loss function that focuses training on the harder samples (i.e. the ones
    that have an uncertain prediction are weighted bigger).

    Arguments:
        gamma - exponent for downweighting easy samples (advised value in paper is 2)
        alpha - weight for class imbalance (same as in weighted crossentropy) Should correspond to the inverse frequency of the foreground pixels.
    '''
    def focal_loss_fixed(y_true, y_pred):
        weight_mask = 10*(alpha * y_pred * (1 - y_true) + (1 - alpha) * (
            1 - y_pred) * y_true)
        return K.mean(
            weight_mask * K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)

    return focal_loss_fixed


def ssim_3d_loss(x, y, **kwargs):
  """ Compute the structural similarity loss between two batches of 3D single channel images

  Parameters
  ----------

  x,y : tensorflow tensors with shape [batch_size,depth,height,width,1] 
    containing a batch of 3D images with 1 channel
  **kwargs : dict
    passed to tf_ssim_3d

  Returns
  -------
  a 1D tensorflow tensor of length batch_size containing the 1 - SSIM for
  every image pair in the batch

  See also
  ----------
  tf_ssim_3d
  """
  return 1 - ssim_3d(x, y, **kwargs)


def generalized_dice_loss(**kwargs):
  """ Generalized dice loss function which is 1 - (generalized dice score)

  Paramters
  ---------

  y_true : tf tensor
    containing the label data. dimensions (n_batch, n0, n1, ...., n_feat)

  y_pred : tf tensor
    containing the predicted data. dimensions (n_batch, n0, n1, ...., n_feat)

  **kwargs : passed to dice_coeff

  Returns
  -------

  A wrapper function that returns 1 - generalized_dice_coeff(y_true, y_pred, **kwargs)
  """

  def generalized_dice_loss_wrapper(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred, **kwargs)

  return  generalized_dice_loss_wrapper


