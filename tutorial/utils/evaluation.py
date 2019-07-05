#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:33:39 2018

@author: teelbo0
"""

import numpy as np
from keras import backend as K


def batch_format_single(y):
    '''
    Makes sure that the given binary mask is in the required format (batch_size, width, height)
    '''
    dims = len(y.shape)

    if dims == 2:
        # the given image is in format (width, height)
        return np.expand_dims(y, axis=0)
    elif dims == 3:
        if y.shape[2] == 1:
            # format (width, height, 1)
            return np.expand_dims(y.reshape(y.shape[:-1]), axis=0)
        else:
            # format (batch_size, width, height)
            return y
    elif dims == 4:
        # format (batch_size, width, height, 1)
        return y.reshape(y.shape[:-1])
    else:
        raise ValueError('The given mask has an invalid format')


def batch_format(y1, y2):
    assert y1.shape == y2.shape

    return (batch_format_single(y1), batch_format_single(y2))


def get_accuracy(y_true, y_pred, threshold=0.5):
    '''
    Returns the 0/1 loss or the accuracy for the given binary masks.
    This corresponds to the number of correctly classified pixels over the total number of pixels.
    '''
    y_true = y_true > 0
    y_pred = y_pred > threshold
    y_true, y_pred = batch_format(y_true, y_pred)

    correct = np.count_nonzero(y_true == y_pred, axis=(1, 2))
    total = y_true.shape[1] * y_pred.shape[2]

    return correct / total


def get_tp(y_true, y_pred, threshold=0.5):
    '''
    Returns the number of true positive pixels for the given binary masks, i.e. the intersection.
    '''
    y_true = y_true > 0
    y_pred = y_pred > threshold
    y_true, y_pred = batch_format(y_true, y_pred)

    tp = np.sum(y_true * y_pred, axis=(1, 2))

    return tp


def get_tn(y_true, y_pred, threshold=0.5):
    '''
    Returns the number of true negative pixels for the given binary masks, i.e. the background, i.e. the intersection of the inverse masks.
    '''
    y_true = y_true > 0
    y_pred = y_pred > threshold
    y_true, y_pred = batch_format(y_true, y_pred)

    tn = np.sum((1 - y_true) * (1 - y_pred), axis=(1, 2))

    return tn


def get_fp(y_true, y_pred, threshold=0.5):
    '''
    Returns the number of false positive pixels for the given binary masks, i.e. the pixels in y_pred that are not in y_true.
    '''
    y_true = y_true > 0
    y_pred = y_pred > threshold
    y_true, y_pred = batch_format(y_true, y_pred)

    fp = np.count_nonzero(np.clip(y_pred.astype(int) - y_true.astype(int), a_min=0, a_max=None), axis=(1, 2))

    return fp


def get_fn(y_true, y_pred, threshold=0.5):
    '''
    Returns the number of false negative pixels for the given binary masks, i.e. the pixels in y_true that are not in y_pred.
    '''
    y_true = y_true > 0
    y_pred = y_pred > threshold
    y_true, y_pred = batch_format(y_true, y_pred)

    fn = np.count_nonzero(np.clip(y_true.astype(int) - y_pred.astype(int), a_min=0, a_max=None), axis=(1, 2))

    return fn


def get_dice(y_true, y_pred, threshold=None):
    '''
    Returns the dice coefficient for the given binary masks.
    Dice = 2*intersection / union
    If threshold is None, the soft dice score is calculated
    '''
    y_true = y_true > 0
    if threshold is not None:
        y_pred = y_pred > threshold
    y_true, y_pred = batch_format(y_true, y_pred)

    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2))
    dice = (2 * intersection + K.epsilon()) / (union + K.epsilon())  # +1 to avoid null division
    return dice


def get_jaccard(y_true, y_pred, threshold=None):
    '''
    Returns the jaccard indexes for the given binary masks.
    JI = intersection / (union - intersection)
    If threshold is None, the soft jaccard score is calculated
    '''
    y_true = y_true > 0
    if threshold is not None:
        y_pred = y_pred > threshold
    y_true, y_pred = batch_format(y_true, y_pred)

    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2))

    jaccard = (intersection + K.epsilon()) / (union - intersection + K.epsilon())  # +1 to avoid null division

    return jaccard


def get_iou(y_true, y_pred):
    '''
    Returns the intersection over union (or IoU) for the given masks.
    This is equivalent to the jaccard index.
    '''
    return get_jaccard(y_true, y_pred)
