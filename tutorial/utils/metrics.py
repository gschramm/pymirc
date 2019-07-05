import numpy as np
from keras import backend as K
import tensorflow as tf


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
    # avoid division by zero by adding 1
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


# aliases
# dice = DICE = dice_coef
# soft_dice = soft_dice_coef
# iou = IOU = IoU
# jaccard = jaccard_index
