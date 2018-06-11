
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
par = {
    'wdir'          :   '/home/mho1/scratch/halo_cnn/',
    'model_name'    :   'halo_cnn1d',
    'data_file'     :   'MDPL2_large_z=0.117.npy',
    
    'bins'          :   6       ,
    'size'          :   None    ,   # None if same size as catalog
    
    'train_only'    :   True    ,   # Only use training data from catalog
    'freq_bin'      :   False   ,   # Bin with equal frequency instead of equal width
    'norm'          :   False   ,   # Normalize input by its standard deviation
    'max_1'         :   False   ,   # X.max() == 1
    
    # Sampling bounds if norm
    'vmax'          :   3.6     ,
    'rmax'          :   1.53    

}

## DATA
print('\nDATA')
# Load and organize

data_path = os.path.join(par['wdir'], 'data')

dat_orig = load_dat(os.path.join(data_path, par['data_file']))

# Subsample
print('\nSUBSAMPLE')

dat = dat_orig[(dat_orig['Mtot']>10**14) & (dat_orig['Mtot']<10**15)]

if par['size'] != None and len(dat) > par['size']:
    dat = np.random.choice(dat, 100, replace=False)

# dat = np.random.choice(dat, 100, replace=False)
print('dat shape: ' + str(dat.shape))



# KDE sample parameters
if par['norm']:
    vmax = par['vmax']
else:
    vmax = dat['vlos'].max()
    par['vmax'] = vmax

print('vmax: ' + str(vmax))




# Generate input data
print('\nGenerate input data')


Xstd = np.ndarray(shape=(len(dat),))
Xkde = np.ndarray(shape = (len(dat), 32))
Xmax = np.ndarray(shape=(len(dat),))

mesh = np.mgrid[-vmax:vmax:32j]
positions = np.vstack([mesh.ravel()])

print('Generating ' + str(len(dat)) + ' KDEs...')
for i in range(len(dat)):
    if i%1000==0: print(str(int(i/1000)) + ' / '+str(int(len(dat)/1000)))
    
    
    Xarr = dat['vlos'][i][:dat['Ngal'][i]]
    
    if par['norm']:
        Xstd[i] = Xarr.std()
        
        Xarr = Xarr/Xstd[i].reshape((1,1))
    
    Xmax[i] = Xarr.max()
    #print (Xarr)
    kde = gaussian_kde(Xarr)
    kdeval = np.reshape(kde(positions).T, mesh.shape)
    
    kdeval /= kdeval.sum()
    #print (kdeval)
    
    Xkde[i,:] = kdeval


print('Max: ' + str(Xarr.max()))

Xkde = Xkde.astype('float32')

if par['max_1']: Xkde /= Xkde.max()

print('\nPREPROCESSING')



if par['norm']:
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
    
    border_dat = m[(dat['intrain']==1)]
    pred_dat = m

if par['freq_bin']:

    border_dat = np.sort(border_dat, axis=0)
    borders = [border_dat[i * len(border_dat)/par['bins']] for i in range(0,par['bins'])]

else:

    borders = np.arange(border_dat.min(), border_dat.max(), (border_dat.max() - border_dat.min())/par['bins'])

par['borders'] = np.append(borders, border_dat.max())

print('borders: ' + str(list(par['borders'])) )

# Assign bins

Y = np.ndarray(shape=(len(pred_dat),1))

for i in range(len(pred_dat)):
    
    Y[i] = par['bins']-1
    
    for j in range(par['bins'] - 1):
        if ((borders[j] <= pred_dat[i]) & (borders[j+1] > pred_dat[i])):
            Y[i] = j
            break



## SAVE
print('\nSAVE')

model_dir = os.path.join(data_path, par['model_name'])

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

print('Writing parameters to file')
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as param_file:
    param_file.write('\n~~~ DATA PARAMETERS ~~~ \n\n')
    for key in par.keys():
        param_file.write(key + ' : ' + str(par[key]) +'\n')
    
    param_file.write('\n\n')

save_dict = {
    'dat_params':   par,
    'X'         :   Xkde,
    'Y'         :   Y,
    'm'         :   pred_dat,
    'in_train'  :   (dat['intrain']==1),
} 



np.save(os.path.join(model_dir, par['model_name'] + '.npy'), save_dict)

print('Data saved.')

## RESULTS
print('\nRESULTS')

if par['norm']:
    f = plt.figure(figsize=(5,5))
    out = binnedplot(m_reg,
                     Mpred_reg, 
                     n=75,
                     percentiles = [2,15,50,85,98]
                    )

    plt.plot(m_reg,m_reg, label='fit')

    plt.xlabel('$\log(M_{200c})$', fontsize=20)
    plt.ylabel('$\log(M_{pred})$', fontsize=20)
    plt.tight_layout()
    f.savefig(os.path.join(model_dir, par['model_name'] + '_regr.pdf'))


f, ax = plt.subplots(figsize=(5,5))

if par['norm']:
    hist_dat = mdiff_reg
else:
    hist_dat = m[(dat['intrain']==1)]

out = histplot(hist_dat, n=75, log=False)

for i in np.append(par['borders'], border_dat.max()):
    ax.axvline(i, color='r')
    
plt.xlabel('$\log(M_{pred}) - \log(M_{200c})$', fontsize=20)
plt.ylabel('$N$', fontsize=20)
plt.tight_layout()
f.savefig(os.path.join(model_dir, par['model_name'] + '_bins.pdf'))


print('Figures saved')
