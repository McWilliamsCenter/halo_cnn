
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
from keras.layers import Conv1D, MaxPooling1D
from keras.constraints import maxnorm


## FUNCTIONS

from tools.matt_tools import *


## PARAMETERS

batch_size = 12
bins = 6
epochs = 50
learning = 0.01

regularization = True



# Files

wdir = '/home/mho1/scratch/halo_cnn/'


model_name = 'halo_cnn1d'
save_dir = os.path.join(wdir, 'saved_models',model_name)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

## FILE SPECIFICATION

log = os.listdir(save_dir)

log = [int(i.split('_')[-1]) for i in log if (i[:-len(i.split('_')[-1])-1]== model_name)]
model_num = max(log) + 1 

## DATA
print('\nDATA')

data_path = os.path.join(wdir, 'data', model_name)

# Load
print('Loading data...')
dat_dict = np.load(os.path.join(data_path, model_name + '.npy'), encoding='latin1').item()

dat_params = dat_dict['dat_params']
X = dat_dict['X']
Y = dat_dict['Y']
mass = dat_dict['m']
in_train = dat_dict['in_train']

print('Data loaded succesfully')



# Train and test
print('\nGENERATING TRAIN/TEST')

X = np.reshape(X, list(X.shape) + [1])


x_train = X[in_train]
y_train = Y[in_train]

x_test = X[~in_train]
y_test = Y[~in_train]

mass_test = mass[~in_train].flatten() 

y_train = keras.utils.to_categorical(y_train, num_classes = bins)
y_test = keras.utils.to_categorical(y_test, num_classes = bins)

## MODEL
print ('\nINITIALIZING MODEL')
model = Sequential()

# """ CONV LAYER 1
model.add(Conv1D(32, 5, input_shape=x_train.shape[1:], padding='same', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Conv1D(32, 5, activation='relu', padding='same', kernel_constraint=maxnorm(3)))

model.add(MaxPooling1D(pool_size=2))

if regularization: model.add(Dropout(0.25))
# """
""" CONV LAYER 2
model.add(Conv1D(32, 5, padding='same', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Conv1D(32, 5, activation='relu', padding='same', kernel_constraint=maxnorm(3)))

model.add(MaxPooling1D(pool_size=2))

if regularization: model.add(Dropout(0.25))
""" 
model.add(Flatten())
# model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(128,  activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(64,  activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.25))
model.add(Dense(bins, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=learning, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print (model.summary())
## TRAIN
print('\nTRAINING MODEL')

validate = np.arange(len(x_test))
np.random.shuffle(validate)
validate = validate[:2000]

hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test[validate], y_test[validate]),
                  shuffle=True,
                  verbose=2)

# Select random test values to displays
if len(x_test)>5:
    print('\nEXAMPLE I/O')
    dispvals = np.arange(len(x_test))
    np.random.shuffle(dispvals)
    dispvals = dispvals[:5]

    print (model.predict(x_test[dispvals]))
    print(y_test[dispvals])

# PLOT AND SAVE


model_dir = os.path.join(save_dir, model_name + '_' + str(model_num))
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
plt.tight_layout()
f.savefig(os.path.join(model_dir, model_name + '_loss.pdf'))


f = plt.figure(figsize=[5,5])
plt.plot(hist.history['acc'],'r',linewidth=3.0)
plt.plot(hist.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.tight_layout()
f.savefig(os.path.join(model_dir, model_name + '_acc.pdf'))


print('Saved training figures\n')
print('Calculating mass error and plotting\n')

borders = dat_dict['dat_params']['borders']
border_mids = [(borders[i] + borders[i+1])/2. for i in range(len(borders)-1)]

pred_bins = model.predict(x_test)
pred_mass = np.dot(pred_bins,border_mids)



f = plt.figure(figsize=[5,5])
#plt.scatter(mass_test,pred_mass, marker='.',alpha=0.5)

one_to_one = np.arange(11)*(mass_test.max() - mass_test.min())/10. + mass_test.min()
plt.plot(one_to_one, one_to_one)

binnedplot(mass_test,pred_mass,n=100, mean=False)
plt.xlabel('M',fontsize=16)
plt.ylabel('$M_{pred}$',fontsize=16)
plt.tight_layout()
f.savefig(os.path.join(model_dir, model_name + '_pred.pdf'))

print('All finished!')
