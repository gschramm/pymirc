from tensorflow import keras

class MNISTCallback(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):
    print('\ncustom callback - learning rate:', logs['lr'])

