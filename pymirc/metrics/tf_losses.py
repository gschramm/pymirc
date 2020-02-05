from keras import backend as K

from utils.metrics import soft_dice_coef, soft_jaccard_index, IoU


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
    Equal to 1 minus the soft dice coefficient defined in utils.metrics
    '''
    return 1 - soft_dice_coef(y_true, y_pred)


def binary_crossentropy(y_true, y_pred):
    '''Redefinition of keras's binary crossentropy'''
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def jaccard(y_true, y_pred):
    '''Equal to 1 minus the jaccard index defined in utils.metrics'''
    return 1 - soft_jaccard_index(y_true, y_pred)


def IoU_loss(y_true, y_pred):
    '''Equal to 1 minus the intersection over union defined in utils.metrics'''
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

