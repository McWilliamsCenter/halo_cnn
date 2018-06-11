
## IMPORTS

import sys
import os
import numpy as np
import numpy.lib.recfunctions as nprf
# import statsmodels.api as sm
from sklearn import linear_model
from scipy.stats import gaussian_kde
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor


## FUNCTIONS

from tools.matt_tools import *


## ML PARAMETERS
par = {
    'batch_size'        :   100,
    'epochs'            :   100,
    'learning'          :   0.001,
    
    'regularization'    :   True,
    'norm_output'       :   True, # Y in [0,1]
    
    'validation'        :   True,
    'crossfold'         :   True
}

## ML Model Declaration
def baseline_model():
    model = Sequential()

    # """ CONV LAYER 1
    model.add(Conv1D(32, 5, input_shape=x_train.shape[1:], padding='same', activation='relu', kernel_constraint=maxnorm(3)))

    model.add(Conv1D(32, 3, activation='relu', padding='same', kernel_constraint=maxnorm(3)))

    model.add(MaxPooling1D(pool_size=2))

    if par['regularization']: model.add(Dropout(0.25))
    
    model.add(Flatten())
    # model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(128,  activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(64,  activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.25))
    model.add(Dense(1))

    opt = keras.optimizers.adam(lr=par['learning'], decay=1e-6)

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=[])

    # cb = keras.callbacks.TensorBoard(log_dir='/home/mho1/temp/', histogram_freq=10,write_images=True)
    
    return model

# Files

wdir = '/home/mho1/scratch/halo_cnn/'

model_name = 'halo_cnn1d_r'
save_dir = os.path.join(wdir, 'saved_models',model_name)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

## FILE SPECIFICATION

log = os.listdir(save_dir)

log = [int(i.split('_')[-1]) for i in log if (i[:-len(i.split('_')[-1])-1]== model_name)]

par['model_num'] = 0 if len(log)==0 else max(log) + 1



## DATA
print('\nDATA')

data_path = os.path.join(wdir, 'data_processed', model_name)



# Load
print('Loading data...')
dat_dict = np.load(os.path.join(data_path, model_name + '.npy'), encoding='latin1').item()

# Unpack data
dat_params = dat_dict['params']
X = dat_dict['X']
Y = dat_dict['Y']
in_train = dat_dict['in_train']
in_test = dat_dict['in_test']
fold = dat_dict['fold']
sigv = dat_dict['sigv']


print('Data loaded succesfully')


par.update(dat_params)


# Preprocessing
X = np.reshape(X, list(X.shape) + [1])

Y_min = Y.min()
Y_max = Y.max()

if par['norm_output']:
    Y -= Y_min
    Y /= (Y_max - Y_min)

par['Y_max'] = Y_max
par['Y_min'] = Y_min

par['fold_max'] = fold.max()

test_folds = [None]

if par['crossfold']==True:
    test_folds = np.arange(par['fold_max']+1)


y_pred = np.zeros(len(in_test))
hist_all = []


t0 = time.time()

for fold_curr in test_folds:
    
    if fold_curr==None:
        print('\n~~~~~NO CROSSFOLD~~~~~\n')
        in_train_curr = in_train==1
        in_test_curr = in_test==1
    else:
        print('\n~~~~~TEST FOLD #' + str(fold_curr) + '~~~~~\n')
        in_train_curr = (in_train==1) & ~(fold==fold_curr)
        in_test_curr = (in_test==1) & (fold == fold_curr)
    
    # Train and test
    print('\nGENERATING TRAIN/TEST DATA')
    
    if par['validation']==True:    
        in_train_ind = np.where(in_train_curr==1)[0]
        
        in_val_curr = np.random.choice(in_train_ind, int(len(in_train_ind)/10), replace=False)
        
        in_val_curr = np.array([(i in in_val_curr) for i in range(len(in_train_curr))])
    else:
        in_val_curr = np.array([False]*len(in_train_curr))


    x_train = X[in_train_curr & ~in_val_curr]
    y_train = Y[in_train_curr & ~in_val_curr]

    x_val = X[in_train_curr & in_val_curr]
    y_val = Y[in_train_curr & in_val_curr]


    # Data augmentation
    print('\nAUGMENTING TRAIN DATA')
    # flip vlos axis
    x_train = np.append(x_train, np.flip(x_train,1),axis=0)
    y_train = np.append(y_train, y_train,axis=0)

    print('\n# of train: '+str(len(y_train)))
    print('\n# of test: ' + str(np.sum(in_test_curr)))

    ## MODEL
    print ('\nINITIALIZING MODEL')

    

    model = baseline_model()

    ## TRAIN
    print('\nTRAINING MODEL')

    hist = model.fit(x_train, y_train,
                      batch_size=par['batch_size'],
                      epochs=par['epochs'],
                      validation_data=(x_val, y_val) if (par['validation']==True) else None,
                      shuffle=True,
                      verbose=2)
                      
    np.put( y_pred, 
            np.where(in_test_curr), 
            model.predict(X[in_test_curr]))
    
    hist_all.append(hist)

t1 = time.time()
print('Training time: ' + str((t1-t0)/60.) + ' minutes')
print('Preparing  output data \n')

sigv_train = sigv[in_train==1]
sigv_test = sigv[in_test==1]

if par['norm_output']:
    y_train = (Y_max - Y_min)*Y[in_train==1] + Y_min
    y_test = (Y_max - Y_min)*Y[in_test==1] + Y_min
    y_regr = y_train
    y_pred= (Y_max - Y_min)*y_pred[in_test==1] + Y_min
else:
    y_train = Y[in_train==1]
    y_test = Y[in_test==1]
    y_regr = y_train
    y_pred = y_pred[in_test==1]
    

y_train = y_train.flatten()
y_test = y_test.flatten()
y_regr = y_regr.flatten()
y_pred = y_pred.flatten()



# PLOT AND SAVE

model_name_save = model_name + '_' + str(par['model_num'])

model_dir = os.path.join(save_dir, model_name_save)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    
with open(os.path.join(wdir, 'data', model_name, 'parameters.txt'),'r') as f:
    with open(os.path.join(model_dir, 'parameters.txt'), 'w') as g:
        
        for line in f:
            g.write(line)
        
        g.write('\n~~~ ML PARAMETERS ~~~ \n')
        g.write('\nbatch_size: ' + str(par['batch_size']))
        g.write('\nepochs: ' + str(par['epochs']))
        g.write('\nlearning: ' + str(par['learning']))
        g.write('\nregularization: ' + str(par['regularization']))
        g.write('\n\n')
        baseline_model().summary(print_fn=lambda x: g.write(x + '\n'))
        
        g.write('\n\n')


# model_path = os.path.join(model_dir, model_name_save + '.h5')
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)




f = plt.figure(figsize=[5,5])

for i in range(len(test_folds)):
    plt.plot(   hist_all[i].history['loss'],
                label=str(test_folds[i]) + ' loss',
                linewidth=3)
                
    if par['validation']==True:
        plt.plot(   hist_all[i].history['val_loss'],
                    label=str(test_folds[i]) + ' val_loss',
                    linewidth=3)
    

plt.legend(fontsize=12)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.tight_layout()
f.savefig(os.path.join(model_dir, model_name_save + '_loss.pdf'))



print('Saved training figures\n')







# sigv_regr = sigv_regr.flatten()


print('Saving output data\n')


save_dict = {
    'params'    :   par,
    
    'sigv_regr' :   sigv_train,
    'y_regr'    :   y_regr,
    'sigv_test' :   sigv_test,
    
    'y_train'   :   y_train,
    'y_test'    :   y_test,
    'y_pred'    :   y_pred
}

np.save(os.path.join(save_dir, 'model_data', model_name_save + '.npy'), save_dict)

print('Output data saved.')







print('All finished!')
