
## IMPORTS

import os
import sys
import numpy as np
import multiprocessing as mp
import pandas as pd
from sklearn import linear_model
from scipy.stats import gaussian_kde
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Cluster, Catalog


## DATA PARAMETERS
par = OrderedDict([
    ('model_name'   ,   'halo_cnn1d_r'),

    ('wdir'         ,   '/home/mho1/scratch/halo_cnn'),
    ('in_folder'    ,   'data_mocks'),
    ('out_folder'   ,   'data_processed'),
    
    ('data_file'    ,   'Rockstar_UM_z=0.117_pure.p'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('shape'        ,   (48,)), # Length of a cluster's ML input array. # of times velocity pdf will be sampled 
    
    ('nfolds'       ,   10 ),
    ('logm_bin'     ,   0.01)

])




## DATA
print('\n~~~~~ LOADING DATA ~~~~~')
# Load and organize

in_path = os.path.join(par['wdir'], par['in_folder'], par['data_file'])

cat = Catalog().load(in_path)

cat.par['vcut'] = 3785.
cat.par['aperature'] = 2.3

# Subsample
print('\n~~~~~ SUBSAMPLING ~~~~~')
    
print('Data length: ' + str(len(cat)))

if par['subsample'] < 1:
    ind = np.random.choice(range(len(cat)), 
                           int(par['subsample']*len(cat)),
                           replace=False
                          )
    cat.prop = cat.prop.iloc[ind]
    cat.gal = cat.gal[ind]

print('Subsampled data length: ' + str(len(cat)))



print('\n~~~~~ DATA CHARACTERISTICS ~~~~~')

print(cat.par)

print('\n~~~~~ PREPROCESSING DATA ~~~~~~')

# Generate input data
print('\nGenerate input data')

# pdfs = np.ndarray(shape = (len(cat), *par['shape']))

mesh = np.mgrid[-cat.par['vcut'] : cat.par['vcut'] : par['shape'][0]*1j]

sample = np.vstack([mesh.ravel()]) # Velocities at fixed intervals. Used to sample velocity pdfs

print('Generating ' + str(len(cat)) + ' KDEs...')

def make_pdf(ind):
    
    # initialize a gaussian kde from galaxy velocities
    kde = gaussian_kde(cat.gal[ind]['vlos'])
    
    # sample kde at fixed intervals
    kdeval = np.reshape(kde(sample).T, mesh.shape)
    
    # normalize input
    kdeval /= kdeval.sum()
    
    return kdeval

with mp.Pool() as pool:
    pdfs = np.array(pool.map(make_pdf, range(len(cat))))

pdfs = pdfs.astype('float32')

print("KDE's generated.")


print('\nConverting masses, sigv to log scale')

masses = np.log10(cat.prop['M200c']).values
masses = masses.reshape((len(masses),1))

sigv = np.log10(cat.prop['sigv']).values
sigv = sigv.reshape((len(masses),1))


## ASSIGN FOLDS
print('\nAssigning test/train in folds...')
fold_ind = pd.Series(np.random.randint(0, par['nfolds'], len(cat.prop['rockstarId'].unique())), 
                  index = cat.prop['rockstarId'].unique())

fold = fold_ind[cat.prop['rockstarId']]

fold_assign = pd.DataFrame(np.zeros(shape=(len(cat),par['nfolds'])), 
                           index = cat.prop.index)

for i in range(par['nfolds']):
    print('fold:',i)
    
    fold_assign.loc[fold_assign.index[fold==i],i] = 2 # Assign test members
    
    log_m = np.log10(cat.prop.loc[fold_assign.index[fold!=i],'M200c'])

    bin_edges = np.arange(log_m.min() * 0.9999, (log_m.max() + par['logm_bin'])*1.0001, par['logm_bin'])
    
    n_per_bin = int(len(log_m)/(10*len(bin_edges)))
    
    for j in range(len(bin_edges)):
        bin_ind = log_m.index[ (log_m >= bin_edges[j])&(log_m < bin_edges[j]+par['logm_bin']) ]
        
        if len(bin_ind) <= n_per_bin:
            fold_assign.loc[bin_ind,i] = 1 # Assign train members
            
        else:
            fold_assign.loc[np.random.choice(bin_ind, n_per_bin), i] = 1

# fold_assign marks test/train within each fold. 0 = None, 1 = train, 2 = test.


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
    
    'pdf'       :   pdfs,
    'mass'      :   masses,
    'sigv'      :   sigv,
    
    'fold_assign':  fold_assign,
} 


np.save(os.path.join(model_dir, par['model_name'] + '.npy'), save_dict)

print('Data saved.')





""" 
## PLOT CHARACTERISTICS
print('\n~~~~~ PLOTTING MASS DIST ~~~~~')


f, ax = plt.subplots(figsize=(5,5))

mass_train = masses[(dat['intrain']==1)]
mass_test = masses[(dat['intest'] == 1)]

_ = matt.histplot(  mass_train, n=75, log=False, 
                    label='train', ax=ax)
_ = matt.histplot(  mass_test, n=75, log=False, 
                    label='test', ax=ax)
    
plt.xlabel('$\log(M_{200c})$', fontsize=20)
plt.ylabel('$N$', fontsize=20)

plt.legend()
plt.tight_layout()
f.savefig(os.path.join(model_dir, par['model_name'] + '_ydist.pdf'))


print('Figures saved')


print('All finished!')

 """

