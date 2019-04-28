
## IMPORTS

import sys
import os
import numpy as np
import time
import pickle

from collections import OrderedDict

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

import tools.matt_tools as matt


## ML PARAMETERS
par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_cnn1d_rP'),

    ('logM_range'   ,   (13., 16.)), # None
    
    ('nbins'        ,   100),

    ('batch_size'   ,   50),
    ('epochs'       ,   20),
    ('learning'     ,   0.001),

    ('validation'   ,   True), # Make a validation set from training data
])


## ML Model Declaration
def baseline_model():
    model = Sequential()

    # """ CONV LAYER 1
    model.add(Conv1D(10, 5, input_shape=x_train.shape[1:], padding='same', activation='relu', kernel_constraint=maxnorm(3)))

    model.add(Conv1D(5, 3, activation='relu', padding='same', kernel_constraint=maxnorm(3)))

    # model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    
    model.add(Dense(128,  activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(64,  activation='relu', kernel_constraint=maxnorm(3)))
    
    model.add(Dense(par['nbins'], activation='softmax'))

    opt = keras.optimizers.adam(lr=par['learning'], decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=[])
    
    return model

# Managing posterior model output
bin_edges = np.linspace(0, 1, par['nbins']+1)
def y_to_bins(y_point):
    y_binned = np.zeros(shape=(len(y_point), par['nbins']), dtype=int)

    for i in range(par['nbins']):
        y_binned[:,i] = (y_point >= bin_edges[i]) & (y_point <= bin_edges[i+1])
        
    return y_binned


bin_centers = [np.mean((bin_edges[i],bin_edges[i+1])) for i in range(par['nbins'])]
def bins_to_y(y_binned):
    y_point = np.dot(y_binned, bin_centers)
    
    return y_point


print('\n~~~~~ MODEL PARAMETERS ~~~~~')
for key in par.keys():
    print(key, ':', str(par[key]))


## DATA
print('\n~~~~~ LOADING PROCESSED DATA ~~~~~')

data_path = os.path.join(par['wdir'], 'data_processed', par['model_name'])
with open(os.path.join(data_path, par['model_name'] + '.p'), 'rb') as f:
    data_dict = pickle.load(f)

# Unpack data
data_params = data_dict['params']
data = data_dict['data']

X = data['pdf'] # pdf input
Y = data['logmass'] # mass output


print('Data loaded succesfully')

data_params.update(par)
par = data_params


print('\n~~~~~ PREPARING X,Y ~~~~~')
X = np.reshape(X, list(X.shape) + [1])

if par['logM_range']==None:
    par['logmass_min'] = Y.min()
    par['logmass_max'] = Y.max()
else:
    par['logmass_min'] = par['logM_range'][0]
    par['logmass_max'] = par['logM_range'][1]


Y -= par['logmass_min']
Y /= (par['logmass_max'] - par['logmass_min'])


par['nfolds'] = data['fold'].max()+ 1 

test_folds = np.arange(par['nfolds'])

print('\n~~~~~ TRAINING ~~~~~')
Y_pred = np.zeros(shape=(len(Y),))
hist_all = []
model_all = []

t0 = time.time()

for fold_curr in test_folds:
    
    # Find relevant clusters in fold
    print('\n~~~~~ TEST FOLD #' + str(fold_curr) + ' ~~~~~\n')
    in_train = data['in_train'] & (data['fold'] != fold_curr)
    in_test = data['in_test'] & (data['fold'] == fold_curr)
    
    # Create train, test samples
    print('\nGENERATING TRAIN/TEST DATA')
    
    if par['validation']==True:    
        in_train_ind = np.where(in_train)[0]
        
        # Choose 1/10 of training data to be validation data
        in_val = np.random.choice(in_train_ind, int(len(in_train_ind)/10), replace=False) 
        
        in_val = np.array([(i in in_val) for i in range(len(in_train))])
    else:
        in_val = np.array([False]*len(in_train))


    x_train = X[in_train & ~in_val]
    y_train = Y[in_train & ~in_val]

    x_val = X[in_train & in_val]
    y_val = Y[in_train & in_val] # Empty if validation==False


    # Data augmentation
    print('AUGMENTING TRAIN DATA')
    # flip vlos axis
    x_train = np.append(x_train, np.flip(x_train,1),axis=0)
    y_train = np.append(y_train, y_train,axis=0)

    print('# of train: '+str(len(y_train)))
    print('# of test: ' + str(np.sum(in_test)))


    # Convert to binned posterior
    print('\nCONVERTING TO BINNED POSTERIOR')
    y_train = y_to_bins(y_train)
    y_val = y_to_bins(y_val)
    
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
                      
    np.put( Y_pred, 
            np.where(in_test), 
            bins_to_y(model.predict(X[in_test])) )
    
    hist_all.append(hist)
    model_all.append(model)

t1 = time.time()
print('\nTraining time: ' + str((t1-t0)/60.) + ' minutes')
print('\n~~~~~ PREPARING RESULTS ~~~~~')

y_test = (par['logmass_max'] - par['logmass_min'])*Y[data['in_test']] + par['logmass_min']

y_pred= (par['logmass_max'] - par['logmass_min'])*Y_pred[data['in_test']] + par['logmass_min']
    
y_test = y_test.flatten()
y_pred = y_pred.flatten()


print('~~~~~ FILE SPECIFICATION ~~~~~')
## FILE SPECIFICATION

save_dir = os.path.join(par['wdir'], 'saved_models', par['model_name'])

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Assign most recent model number
log = os.listdir(save_dir)

log = [int(i.split('_')[-1]) for i in log if (i[:-len(i.split('_')[-1])-1]== par['model_name'])]

par['model_num'] = 0 if len(log)==0 else max(log) + 1


print('~~~~~ SAVING PARAMETERS ~~~~~')

model_name_save = par['model_name'] + '_' + str(par['model_num'])

model_dir = os.path.join(save_dir, model_name_save)
model_fold_dir = os.path.join(save_dir, model_name_save,'models')

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    os.makedirs(model_fold_dir)

with open(os.path.join(model_dir, 'parameters.txt'), 'w') as param_file:
    param_file.write('\n~~~ DATA, ML PARAMETERS ~~~ \n\n')
    for key in par.keys():
        param_file.write(key + ' : ' + str(par[key]) +'\n')
    
    param_file.write('\n\n')
    
    baseline_model().summary(print_fn=lambda x: param_file.write(x + '\n'))
    
    param_file.write('\n\n')

print('~~~~~ SAVING MODELS ~~~~~')
for i in range(len(test_folds)):
    model_path = os.path.join(model_fold_dir, 'fold_' + str(i) + '.h5')
    model_all[i].save(model_path)
#print('Saved trained model at %s ' % model_path)

print('~~~~~ SAVING LOSS CURVES ~~~~~')
loss_dat = np.zeros(shape=(len(test_folds), par['epochs']), dtype = [('train','<f4'), ('val','<f4')] )

loss_dat['train'] = np.array([hist_all[i].history['loss'] for i in range(len(test_folds)) ])
if par['validation']==True: loss_dat['val'] = np.array([hist_all[i].history['val_loss'] for i in range(len(test_folds)) ])

np.save(os.path.join(model_dir, model_name_save + '_loss.npy'), loss_dat)

print('~~~~~ PLOTTING ~~~~~')

f = plt.figure(figsize=[3,3])

for i in [0]:#range(len(test_folds)):
    plt.plot(   hist_all[i].history['loss'],
                label='training')
                
    if par['validation']==True:
        plt.plot(   hist_all[i].history['val_loss'],
                    label='validation')
    

plt.legend(fontsize=8)
plt.xlabel('Epochs ',fontsize=10)
plt.ylabel('Loss',fontsize=10)
plt.tight_layout()
f.savefig(os.path.join(model_dir, model_name_save + '_loss.pdf'))



print('Saved training figures\n')



print('Saving output data\n')


save_dict = {
    'params'    :   par,

    'logmass_test'    :   y_test,
    'logmass_pred'    :   y_pred
}

np.save(os.path.join(model_dir, model_name_save + '.npy'), save_dict)

print('Output data saved.')







print('All finished!')
