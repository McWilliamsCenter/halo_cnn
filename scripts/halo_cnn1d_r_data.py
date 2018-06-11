
## IMPORTS

import os
import numpy as np
from sklearn import linear_model
from scipy.stats import gaussian_kde
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## FUNCTIONS

from tools.matt_tools import *


## DATA PARAMETERS
par = OrderedDict([

    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_cnn1d_r'),
    ('data_file'    ,   'UM_z=0.117_med_reduced.npy'),
    
    ('subsample'    ,   0.5 ), # Fraction by which to randomly subsample data
    
    ('norm'         ,   False),
    ('max_1'        ,   False),
    ('new_train'    ,   False),
    
    ('vmax'         ,   3.6),
    ('rmax'         ,   1.53),
    
    ('shape'        ,   48)
])

par = {
    'wdir'          :   '/home/mho1/scratch/halo_cnn/',
    'model_name'    :   'halo_cnn1d_r',
    # 'data_file'     :   'MDPL2_large_z=0.117_reduced.npy',
    # 'data_file'     :   'UM_z=0.117_reduced.npy',
    'data_file'     :   'UM_z=0.117_med_reduced.npy',
    
    'size'          :   None    ,   # None if same size as catalog
    'subsample'     :   0.5,
    
    'norm'          :   False   ,   # Normalize input by its standard deviation
    'max_1'         :   False   ,   # X.max() == 1
    'new_train'     :   False   ,   # Reassign train datapoints
    
    # Sampling bounds if norm
    'vmax'          :   3.6     ,
    'rmax'          :   1.53    ,
    'shape'         :   48

}


## DATA
print('\n~~~~~ LOADING DATA ~~~~~')
# Load and organize

raw_path = os.path.join(par['wdir'], 'data_reduced')

print('\nData file: ' + par['data_file'] + '\n')

dat_orig = np.load(os.path.join(raw_path, par['data_file']))
print(str(sys.getsizeof(dat_orig)/10.**9) + ' GB\n')

# Subsample
print('\nSUBSAMPLE')
    
print('Original data length: ' + str(len(dat_orig)))
dat = np.random.choice( dat_orig, 
                        int(par['subsample']* len(dat_orig)),
                        replace = False)    
print('New data length: ' + str(len(dat)))
    
    # dat = dat_orig[(dat_orig['Mtot']>10**13.5) & (dat_orig['Mtot']<10**15.25)]

# Reassign train datapoints
if par['new_train']:
    print('\nREASSIGN TRAINING DATA')

    intrvl = 0.01
    num_train = 4000

    dat['intrain']=0

    logM = np.log10(dat['Mtot'])

    kde = gaussian_kde(logM) # get a probability distribution for logM

    sample = np.arange(logM.min(),
                       logM.max(),
                       intrvl
                      )
    pdf = kde(sample)
    pdf /= pdf.sum()
    
    pdf = 1./pdf # invert it
    pdf /= pdf.sum()

    # sample pdf at each datapoint
    p = np.ndarray(shape=len(logM))

    for i in range(len(logM)):
        p[i] = pdf[int((logM[i] - logM.min())/intrvl)] 

    p/=p.sum()


    new_logM_indices = np.random.choice(np.arange(len(logM)), num_train, replace=False, p=p) # use the probabilities to choose an even mass profile
    
    
    dat['intrain'][new_logM_indices]=1


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
Xkde = np.ndarray(shape = (len(dat), par['shape']))
Xmax = np.ndarray(shape=(len(dat),))

mesh = np.mgrid[-vmax : vmax : par['shape']*1j]

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


# Bin data
print('\nBIN DATA')

m = np.log10(dat['Mtot'])
m = m.reshape((len(m),1))

Y = m

# Power law predictions
print('\nPOWER LAW PREDICTIONS')
sigv = np.log10(dat['sigmav'])
sigv = sigv.reshape((len(m),1))


## SAVE
print('\nSAVE')

model_dir = os.path.join(par['wdir'], 'data_processed', par['model_name'])

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

print('Writing parameters to file')
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as param_file:
    param_file.write('\n~~~ DATA PARAMETERS ~~~ \n\n')
    for key in par.keys():
        param_file.write(key + ' : ' + str(par[key]) +'\n')
    
    param_file.write('\n\n')

save_dict = {
    'params'    :   par,
    'X'         :   Xkde,
    'Y'         :   Y,
    'sigv'      :   sigv,
    'in_train'  :   (dat['intrain']==1),
    'in_test'   :   (dat['intest']==1),
    'fold'      :   dat['fold']
} 


np.save(os.path.join(model_dir, par['model_name'] + '.npy'), save_dict)

print('Data saved.')

## RESULTS
print('\nRESULTS')


f, ax = plt.subplots(figsize=(5,5))

hist_dat = m[(dat['intrain']==1)]

_ = histplot(hist_dat, n=75, log=False, ax=ax)
    
plt.xlabel('$\log(M_{pred}) - \log(M_{200c})$', fontsize=20)
plt.ylabel('$N$', fontsize=20)
plt.title('Training Data Distribution', fontsize=20)
plt.tight_layout()
f.savefig(os.path.join(model_dir, par['model_name'] + '_ydist.pdf'))


print('Figures saved')


print('All finished!')
