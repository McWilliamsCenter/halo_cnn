
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


## PARAMETERS

batch_size = 32
bins = 6
epochs = 100

# KDE sample parameters
vmax = 5.33
xmax = 3.37
ymax = 3.31

print ('xmax, ymax, vmax: ', xmax, ymax, vmax)


# Files

wdir = '/home/mho1/2_pylon2/'

data_path = os.path.join(wdir, 'data')
model_name = 'halo_cnn3d'


## FUNCTIONS

from matt_tools import *


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
    
    # Xarr -= Xarr.mean(axis=1).reshape((3,1)) # shift to 0 mean
    
    Xstd[i,:] = Xarr.std(axis=1)
    
    Xarr = Xarr/Xstd[i,:].reshape((3,1))
    
    # Xmax[i,:] = Xarr.max(axis=1)
    
    kde = gaussian_kde(Xarr)
    kdeval = np.reshape(kde(positions).T, mesh[0].shape)

    kdeval /= kdeval.sum()
    
    Xkde[i,:,:,:] = kdeval

Xkde = Xkde.astype('float32')
# Xkde /= Xkde.max()

X = np.log10(Xstd)


# Regression

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X,m)


print ('regr coef: ', regr.coef_)
print ('regr intercept: ', regr.intercept_)
print ('regr R^2: ', regr.score(X,m))

Mpred = regr.predict(X)

# Y = m > Mpred
# Y = Y.astype('int')

# Bin data
mdiff = (Mpred - m)
print(mdiff.mean())
print(mdiff.std())

mdiff = (mdiff - mdiff.mean())
mdiff_s = np.sort(mdiff, axis=0)


borders = [mdiff_s[i * len(mdiff_s)/bins] for i in range(0,bins)]

print('borders: ', borders)

Y = np.ndarray(shape=(len(mdiff),1))

for i in range(len(mdiff)):
    
    Y[i] = bins-1
    
    for j in range(bins - 1):
        if ((borders[j] <= mdiff[i]) & (borders[j+1] > mdiff[i])):
            Y[i] = j
            break
            
## SAVE
model_dir = os.path.join(data_path,model_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

np.save(os.path.join(model_dir, model_name + '_x.npy'), Xkde)
np.save(os.path.join(model_dir, model_name + '_y.npy'), Y)

print('Data saved.')


## RESULTS
f = plt.figure(figsize=(7,7))
out = binnedplot(m,
                 Mpred, 
                 n=75,
                 percentiles = [2,15,50,85,98]
                )

plt.plot(m,m, label='fit')

plt.xlabel('$\log(M_{200c})$', fontsize=20)
plt.ylabel('$\log(M_{pred})$', fontsize=20)
f.savefig(os.path.join(model_dir, model_name + '_regr.pdf'))


f, ax = plt.subplots(figsize=(7,7))

out = histplot(mdiff, n=75, log=False)

for i in borders + [mdiff_s[-1]]:
    ax.axvline(i, color='r')
    
plt.xlabel('$\log(M_{pred}) - \log(M_{200c})$', fontsize=20)
plt.ylabel('$N$', fontsize=20)
f.savefig(os.path.join(model_dir, model_name + '_bins.pdf'))


print('Figures saved')
