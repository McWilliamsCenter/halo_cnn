
## IMPORTS

import os
import numpy as np
from sklearn import linear_model
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## FUNCTIONS

from tools.matt_tools import *
from tools.cnn_tools import *


## PARAMETERS

bins = 6

train_only = True
freq_bin = False
small = False


# Normalized KDE
normKDE = False

# X.max() == 1 
max1 = True

# X[X<X_to_0] = 0
X_to_0 = 0.1



vmax = 3.6
rmax = 1.53


# Files

wdir = '/home/mho1/scratch/halo_cnn/'

data_path = os.path.join(wdir, 'data')
model_name = 'halo_cnn2d'
data_file = "MDPL2_large_z=0.117.npy"



## DATA
print('\nDATA')
# Load and organize

dat_orig = load_dat(os.path.join(data_path, data_file))

# Subsample
print('\nSUBSAMPLE')

dat = dat_orig[(dat_orig['Mtot']>10**14) & (dat_orig['Mtot']<10**15)]

if train_only:
    dat = dat[(dat['intrain']==1)]
if small and len(dat) > 100:
    dat = np.random.choice(dat, 100, replace=False)

# dat = np.random.choice(dat, 100, replace=False)
print('dat shape: ' + str(dat.shape))



# KDE sample parameters
if not normKDE:
    vmax = dat['vlos'].max()
    rmax = dat['Rproj'].max()

print('vmax: ' + str(vmax))
print('rmax: ' + str(rmax))




# Generate input data
print('\nGenerate input data')


Xstd = np.ndarray(shape=(len(dat),2))
Xkde = np.ndarray(shape = (len(dat), 32, 32))
Xmax = np.ndarray(shape=(len(dat),2))

mesh = np.mgrid[-vmax:vmax:32j, 0:rmax:32j]
positions = np.vstack([mesh[0].ravel(), mesh[1].ravel()])

print('Generating ' + str(len(dat)) + ' KDEs...')
for i in range(len(dat)):
    if i%1000==0: print (i)
    
    Xarr = np.ndarray(shape=(2,dat['Ngal'][i]))
    
    Xarr[0,:] = dat['vlos'][i][:dat['Ngal'][i]]
    Xarr[1,:] = dat['Rproj'][i][:dat['Ngal'][i]]
    
    if normKDE:
        Xstd[i,0] = Xarr[0,:].std()
        Xstd[i,1] = np.sqrt((Xarr[1,:]**2).mean())
        
        Xarr = Xarr/Xstd[i,:].reshape((2,1))
    
    Xmax[i,:] = Xarr.max(axis=1)
    
    kde = gaussian_kde(Xarr)
    kdeval = np.reshape(kde(positions).T, mesh[0].shape)
    
    kdeval /= kdeval.sum()

    
    Xkde[i,:,:] = kdeval


print('Max: ' + str(Xarr.max(axis=1)))

Xkde = Xkde.astype('float32')

if max1: Xkde /= Xkde.max()

print('\nPREPROCESSING')


"""
Xkde = np.log10(Xkde)
Xkde[Xkde == -np.inf] = Xkde.min()
Xkde += Xkde.min()
Xkde /= Xkde.max()


X = Xkde
X_sort = np.sort(X.flatten())
for i in range(len(X)):
    for j in range(len(X[i])):
        for k in range(len(X[i][j])):
            X[i,j,k] = np.random.choice(np.where(X_sort == X[i,j,k])[0])

X = X.astype('float32')
X/= X.max()
Xkde = X"""

if normKDE:
    # Regression
    print('\nREGRESSION')

    X_reg = np.log10(Xstd[(dat['intrain']==1)])
    m_reg = np.log10(dat['Mtot'][(dat['intrain']==1)])
    m_reg = m_reg.reshape((len(m_reg),1))

    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(X_reg,m_reg)


    print('regr coef: ' +  str(regr.coef_))
    print('regr intercept: ' + str(regr.intercept_))
    print('regr R^2: ' + str(regr.score(X_reg,m_reg)))

    Mpred_reg = regr.predict(X_reg)

    # Y = m > Mpred
    # Y = Y.astype('int')

    # Bin data
    print('\nBIN DATA')

    mdiff_reg = (Mpred_reg - m_reg)/ Mpred_reg
    print('Mdiff mean: ' + str(mdiff_reg.mean()))
    print('Mdiff std: ' + str(mdiff_reg.std()))

    # mdiff = (mdiff - mdiff.mean())
    mdiff_reg = np.sort(mdiff_reg, axis=0)
    
    m = np.log10(dat['Mtot'])
    m = m.reshape((len(m),1))
    Mpred = regr.predict(Xstd)
    mdiff = (Mpred - m)/Mpred
    
    border_dat = mdiff_reg
    pred_dat = mdiff

else:
    # Bin data
    print('\nBIN DATA')
    
    m = np.log10(dat['Mtot'])
    m = m.reshape((len(m),1))
    
    border_dat = m
    pred_dat = m

if freq_bin:

    border_dat = np.sort(border_dat, axis=0)
    borders = [border_dat[i * len(border_dat)/bins] for i in range(0,bins)]

else:

    borders = np.arange(border_dat.min(), border_dat.max(), (border_dat.max() - border_dat.min())/bins)


print('borders: ' + str(list(borders)) )

# Assign bins

Y = np.ndarray(shape=(len(pred_dat),1))

for i in range(len(pred_dat)):
    
    Y[i] = bins-1
    
    for j in range(bins - 1):
        if ((borders[j] <= pred_dat[i]) & (borders[j+1] > pred_dat[i])):
            Y[i] = j
            break

# i = np.random.randint(len(Xkde),size=5)
# Xkde = np.take(Xkde,i,0)
# Y = np.take(Y,i,0)

## SAVE
print('\nSAVE')

model_dir = os.path.join(data_path,model_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

print('Writing parameters to file')
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as params:
    params.write('\n~~~ DATA PARAMETERS ~~~ \n\n')
    params.write('wdir: ' + wdir + '\n')
    params.write('model_name: ' + model_name+ '\n')
    params.write('data_file: ' + data_file+ '\n')
    
    params.write('bins: ' + str(bins)+ '\n')
    params.write('train_only: ' + str(train_only)+ '\n')
    params.write('freq_bin: ' + str(freq_bin)+ '\n')
    params.write('small: ' + str(small)+ '\n')
    params.write('normKDE: ' + str(normKDE)+ '\n')
    
    params.write('\nvmax: ' + str(vmax)+ '\n')
    params.write('rmax: ' + str(rmax)+ '\n')
    params.write('\n\n')


np.save(os.path.join(model_dir, model_name + '_x.npy'), Xkde)
np.save(os.path.join(model_dir, model_name + '_y.npy'), Y)

print('Data saved.')

## RESULTS
print('\nRESULTS')

if normKDE:
    f = plt.figure(figsize=(5,5))
    out = binnedplot(m_reg,
                     Mpred_reg, 
                     n=75,
                     percentiles = [2,15,50,85,98]
                    )

    plt.plot(m_reg,m_reg, label='fit')

    plt.xlabel('$\log(M_{200c})$', fontsize=20)
    plt.ylabel('$\log(M_{pred})$', fontsize=20)
    f.savefig(os.path.join(model_dir, model_name + '_regr.pdf'))


f, ax = plt.subplots(figsize=(5,5))

if normKDE:
    hist_dat = mdiff_reg
else:
    hist_dat = m

out = histplot(hist_dat, n=75, log=False)

for i in np.append(borders, border_dat.max()):
    ax.axvline(i, color='r')
    
plt.xlabel('$\log(M_{pred}) - \log(M_{200c})$', fontsize=20)
plt.ylabel('$N$', fontsize=20)
f.savefig(os.path.join(model_dir, model_name + '_bins.pdf'))


print('Figures saved')
