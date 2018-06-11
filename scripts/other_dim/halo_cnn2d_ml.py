
## IMPORTS

import sys
import os
import numpy as np
import numpy.lib.recfunctions as nprf
from sklearn import linear_model
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import maxnorm


## FUNCTIONS

from tools.matt_tools import *


## PARAMETERS

batch_size = 12
bins = 6
epochs = 70
learning = 0.01

regularization = False



# Files

wdir = '/home/mho1/scratch/halo_cnn/'

save_dir = os.path.join(wdir, 'saved_models')
model_name = 'halo_cnn2d'

data_path = os.path.join(wdir, 'data', model_name)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

## FILE SPECIFICATION

log = os.listdir(save_dir)
model_num = int(np.sum([i[:-len(i.split('_')[-1])-1]== model_name for i in log]))

model_dir = os.path.join(save_dir, model_name + '_' + str(model_num))

## DATA
print('\nDATA')

# Load
print('Loading data...')
Xkde = np.load(os.path.join(data_path, model_name + '_x.npy'))
Y = np.load(os.path.join(data_path, model_name + '_y.npy'))

print('Data loaded succesfully')



# Train and test
print('\nGENERATING TRAIN/TEST')

Xkde = np.reshape(Xkde, list(Xkde.shape) + [1])

indices = list(range(len(Xkde)))
test = np.random.choice(indices, int(len(Xkde)/10), replace=False)
train = [i for i in range(len(Xkde)) if i not in test]

x_train = Xkde[train,:,:]
y_train = Y[train]

x_test = Xkde[test,:,:]
y_test = Y[test]

y_train = keras.utils.to_categorical(y_train, num_classes = bins)
y_test = keras.utils.to_categorical(y_test, num_classes = bins)

## MODEL
print ('\nINITIALIZING MODEL')
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=x_train.shape[1:], padding='same', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(3, 3)))

if regularization: model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.25))
model.add(Dense(bins))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=learning, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print (model.summary())
## TRAIN
print('\nTRAINING MODEL')
hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  verbose=2)

"""print (model.predict(x_test))
print(y_test)"""

# PLOT AND SAVE
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    
with open(os.path.join(wdir, 'data', model_name, 'parameters.txt'),'r') as f:
    with open(os.path.join(model_dir, 'parameters.txt'), 'w') as g:
        
        for line in f:
            g.write(line)
        
        g.write('\n~~~ ML PARAMETERS ~~~ \n')
        g.write('\nbatch_size: ' + str(batch_size))
        g.write('\nbins: ' + str(bins))
        g.write('\nepochs: ' + str(epochs))
        g.write('\nlearning: ' + str(learning))
        g.write('\nregularization: ' + str(regularization))
        g.write('\n\n')
        model.summary(print_fn=lambda x: g.write(x + '\n'))
        
        g.write('\n\n')


model_path = os.path.join(model_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

f = plt.figure(figsize=[5,5])
plt.plot(hist.history['loss'],'r',linewidth=3.0)
plt.plot(hist.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
f.savefig(os.path.join(model_dir, model_name + '_loss.pdf'))


f = plt.figure(figsize=[5,5])
plt.plot(hist.history['acc'],'r',linewidth=3.0)
plt.plot(hist.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
f.savefig(os.path.join(model_dir, model_name + '_acc.pdf'))

print('Saved figures')

