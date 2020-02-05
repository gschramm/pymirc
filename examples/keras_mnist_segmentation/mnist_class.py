# example from https://www.tensorflow.org/guide/keras/train_and_evaluate

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as py

batch_size    = 512
epochs        = 150
learning_rate = 3e-3
nval          = 10000

# Load a toy dataset for the sake of this example
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = np.expand_dims(x_train,-1).astype('float32') / 255
x_test  = np.expand_dims(x_test,-1).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-nval:]
y_val = y_train[-nval:]
x_train = x_train[:-nval]
y_train = y_train[:-nval]

input_shape = x_train.shape[1:]
num_classes = np.unique(y_train).shape[0] 

#------------------------------
#------------------------------
#------------------------------

# define the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape = input_shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# Specify the training configuration (optimizer, loss, metrics)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = learning_rate), 
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

#-----------------------------------------------------------------
# train the model

# define a callback that reduces the learning rate

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                              patience=10, min_lr = 1e-6)

history = model.fit(x_train, 
                    y_train,
                    batch_size      = batch_size,
                    epochs          = epochs,
                    validation_data = (x_val, y_val),
                    callbacks       = [reduce_lr])

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
predictions_probs = model.predict(x_test)
predictions       = np.argmax(predictions_probs,1)
predictions_prob  = predictions_probs.max(1)


#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------

# show some results
py.rcParams['axes.titlesize'] = 'x-small'
py.rcParams['image.cmap']     = 'Greys'

#------------------------------------
# show the first 100 wrong predictions
inds_wrong  = np.where(predictions != y_test)
n_wrong     = inds_wrong[0].shape[0]
wrong_count = np.bincount(y_test[inds_wrong].astype(int)) 
ncols = int(min(10, np.ceil(np.sqrt(n_wrong))))
nrows = int(min(10, np.ceil(n_wrong/ncols)))

fig, ax  = py.subplots(nrows, ncols, figsize=(9, 9))

for ii, axx in enumerate(ax.flatten()):
  if ii < n_wrong:
    i = inds_wrong[0][ii]
    axx.imshow(x_test[i].reshape(28,28))
    axx.set_title('gt' + str(int(y_test[i])) + ' ' + str(predictions[i]) + ' ' + str(round(predictions_prob[i],3)))
  axx.set_axis_off()

fig.tight_layout()
fig.show()

#------------------------------------
# show the first 100 wrong predictions
inds_low  = np.where(predictions_prob < 0.6)
n_low     = inds_low[0].shape[0]
low_count = np.bincount(y_test[inds_low].astype(int)) 
ncols = int(min(10, np.ceil(np.sqrt(n_low))))
nrows = int(min(10, np.ceil(n_low/ncols)))

fig2, ax2  = py.subplots(nrows, ncols, figsize=(9, 9))

for ii, axx in enumerate(ax2.flatten()):
  if ii < n_low:
    i = inds_low[0][ii]
    axx.imshow(x_test[i].reshape(28,28))
    axx.set_title('gt' + str(int(y_test[i])) + ' ' + str(predictions[i]) + ' ' + str(round(predictions_prob[i],3)))
  axx.set_axis_off()

fig2.tight_layout()
fig2.show()

#------------------------------------
# plot the loss functions

py.rcParams['axes.titlesize'] = 'medium'
fig3, ax3 = py.subplots(1,2, figsize = (8,4))
ax3[0].plot(history.history['loss'], label = 'train')
ax3[0].plot(history.history['val_loss'], label = 'val')
ax3[0].legend()
ax3[0].set_title('loss')
ax3[0].grid(ls = ':')
ax3[0].set_ylim(None,1.02*max(history.history['val_loss']))
ax3[1].plot(history.history['sparse_categorical_accuracy'])
ax3[1].plot(history.history['val_sparse_categorical_accuracy'])
ax3[1].set_title('sparse_categorical_accuracy')
ax3[1].grid(ls = ':')
ax3[1].set_ylim(0.98*min(history.history['val_sparse_categorical_accuracy']),None)
fig3.tight_layout()
fig3.show()
