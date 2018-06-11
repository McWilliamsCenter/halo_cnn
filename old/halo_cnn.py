
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
from keras.layers import Conv3D, MaxPooling3D

## PARAMETERS

batch_size = 32
num_classes = 2
epochs = 100

# KDE sample parameters
vmax = 5.33
xmax = 3.37
ymax = 3.31

print ('xmax, ymax, vmax: ', xmax, ymax, vmax)


# Files

wdir = '/home/mho1/2_pylon2/'

data_path = os.path.join(wdir, 'data')
save_dir = wdir + 'saved_models'
model_name = 'halo_cnn1'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

## FUNCTIONS

def binnedplot(X,Y, n=10, percentiles = [15,50,85], mean=True, label='', ax = None):
    if n=='auto':
        n = len(X)/1000

    increment = (X.max() - X.min())/n
    i = X.min()

    x = []
    y = []
    
    y_percent = np.ndarray((0,len(percentiles)))
    
    err = []
    while i < X.max():
        bindata = Y[(X>=i) & (X<(i+increment))]
        
        i+=increment
        
        if len(bindata) == 0: continue
            
        x.append(i-increment/2)
        
        y.append(bindata.mean())
        
        y_p = np.percentile(bindata,percentiles)
        y_percent = np.append(y_percent, [y_p],axis=0)
        
        err.append(bindata.std())

        
    if mean: 
        if ax is None: plt.plot(x,y, label=label+'mean')
        else: ax.plot(x,y, label=label+'mean')
    y_percent = np.swapaxes(y_percent,0,1)
    
    for i in range(len(percentiles)):
        if ax is None:
            plt.plot(x, y_percent[i],label=label+str(percentiles[i]))
        else:
            ax.plot(x, y_percent[i],label=label+str(percentiles[i]))

    return x,y,y_percent, err





## FILE SPECIFICATION

log = os.listdir(save_dir)
model_num = int(np.sum([i[:-len(i.split('_')[-1])-1]== model_name for i in log]))

model_dir = os.path.join(save_dir, model_name + '_' + str(model_num))

## DATA

# Load and organize
print('Loading data...')
dat_orig = np.load(os.path.join(data_path, "MDPL2_large_z=0.117.npy"))
print('Data loaded successfully')

keep = ['Mtot', 'rotation', 'fold', 'Ngal', 'vlos', 'Rproj',
        'name', 'intest', 'intrain', 'redshift','sigmav', 'xyproj'
       ]
print(sys.getsizeof(dat_orig)/10.**9, ' GB')
temp = nprf.drop_fields(dat_orig, [i for i in dat_orig.dtype.names if i not in keep])
print( sys.getsizeof(temp)/10.**9, ' GB')
del(dat_orig)

dat_orig = temp


# Subsample

dat = dat_orig[(dat_orig['Mtot']>10**14) & (dat_orig['Mtot']<10**15) & (dat_orig['intrain']==1)]

# dat = np.random.choice(dat, 100, replace=False)
print (dat.shape)


# Generate input data
m = np.log10(dat['Mtot'])
m = m.reshape((len(m),1))

Xstd = np.ndarray(shape=(len(dat),3))
Xkde = np.ndarray(shape = (len(dat), 32, 32, 32))
#Xmax = np.ndarray(shape=(len(dat),3))

mesh = np.mgrid[-vmax:vmax:32j, -xmax:xmax:32j, -ymax:ymax:32j]
positions = np.vstack([mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()])

print('Generating KDEs')
for i in range(len(dat)):
    if i%500==0: print (i)
    
    Xarr = np.ndarray(shape=(3,dat['Ngal'][i]))
    
    Xarr[0,:] = dat['vlos'][i][:dat['Ngal'][i]]
    Xarr[1:,:] = np.array(list(zip(*dat['xyproj'][i][:dat['Ngal'][i]])))
    
    Xarr = Xarr - Xarr.mean(axis=1).reshape((3,1)) # shift to 0 mean
    
    Xstd[i,:] = Xarr.std(axis=1)
    
    Xarr = Xarr/Xstd[i,:].reshape((3,1))
    
    # Xmax[i,:] = Xarr.max(axis=1)
    
    kde = gaussian_kde(Xarr)
    kdeval = np.reshape(kde(positions).T, mesh[0].shape)
    
    kdeval = kdeval/kdeval.max()
    
    Xkde[i,:,:,:] = kdeval

Xkde = Xkde.astype('float32')


# Regression

X = np.log10(Xstd)

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X,m)


print ('regr coef: ', regr.coef_)
print ('regr intercept: ', regr.intercept_)
print ('regr R^2: ', regr.score(X,m))

Mpred = regr.predict(X)

Y = m > Mpred
Y = Y.astype('int')


# Train and test

Xkde = np.reshape(Xkde, list(Xkde.shape) + [1])

indices = list(range(len(dat)))
test = np.random.choice(indices, int(len(dat)/10), replace=False)
train = [i for i in range(len(dat)) if i not in test]

x_train = Xkde[train,:,:,:]
y_train = Y[train]

x_test = Xkde[test,:,:,:]
y_test = Y[test]



## MODEL

model = Sequential()
model.add(Conv3D(32, (5, 5, 5), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv3D(16, (3, 3, 3)))
model.add(Activation('relu'))
model.add(Conv3D(16, (3, 3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print (model.summary())

hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  verbose=2)


# PLOT AND SAVE
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

f = plt.figure(figsize=[8,6])
plt.plot(hist.history['loss'],'r',linewidth=3.0)
plt.plot(hist.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
f.savefig(os.path.join(model_dir,'training_loss.pdf'))

f = plt.figure(figsize=[8,6])
plt.plot(hist.history['acc'],'r',linewidth=3.0)
plt.plot(hist.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
f.savefig(os.path.join(model_dir,'training_acc.pdf'))


f = plt.figure(figsize=(7,7))
out = binnedplot(m,
                 Mpred, 
                 n=75,
                 percentiles = [2,15,50,85,98]
                )

plt.plot(m,m, label='fit')

plt.xlabel('$\log(M_{200c})$', fontsize=20)
plt.ylabel('$\log(M_{pred})$', fontsize=20)
f.savefig(os.path.join(model_dir,'regr.pdf'))
