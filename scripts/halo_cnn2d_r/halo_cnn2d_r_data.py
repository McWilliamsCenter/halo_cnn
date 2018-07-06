
## IMPORTS

import os
import sys
import numpy as np
import multiprocessing as mp
import pandas as pd
import pickle

from sklearn import linear_model
from scipy.stats import gaussian_kde
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Catalog


## DATA PARAMETERS
par = OrderedDict([
    ('model_name'   ,   'halo_cnn2d_r'),

    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('in_folder'    ,   'data_mocks'),
    ('out_folder'   ,   'data_processed'),
    
    ('data_file'    ,   'Rockstar_UM_z=0.117_contam_rot10.p'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('shape'        ,   (48,48)), # Length of a cluster's ML input array. # of times velocity pdf will be sampled 
    
    ('bandwidth'    ,   'avg_scott'), # bandwidth used for gaussian kde. Can be scalar, 'scott','silverman', or 'avg_scott'
    
    ('nfolds'       ,   10),
    ('logm_bin'     ,   0.01),
    ('mbin_frac'    ,   0.025)

])
# For running
n_proc = 4



## DATA
print('\n~~~~~ LOADING DATA ~~~~~')
# Load and organize

in_path = os.path.join(par['wdir'], par['in_folder'], par['data_file'])

cat = Catalog().load(in_path)

if (cat.par['vcut'] is None) & (cat.par['aperture'] is None):
    print('Pure catalog...')
    cat.par['vcut'] = 3785.
    cat.par['aperture'] = 2.3

# Subsample
print('\n~~~~~ SUBSAMPLING ~~~~~')
    
print('Data length: ' + str(len(cat)))

if par['subsample'] < 1:
    ind = np.random.choice(range(len(cat)), 
                           int(par['subsample']*len(cat)),
                           replace=False
                          )
    cat = cat[ind]

print('Subsampled data length: ' + str(len(cat)))



print('\n~~~~~ DATA CHARACTERISTICS ~~~~~')
for key in cat.par.keys():
    print(key,':', cat.par[key])


print('\n~~~~~ ASSIGNING FOLDS ~~~~~~')

## ASSIGN FOLDS
print('\nAssigning test/train in folds...')
fold_ind = pd.Series(np.random.randint(0, par['nfolds'], len(cat.prop['rockstarId'].unique())), 
                  index = cat.prop['rockstarId'].unique())

fold = fold_ind[cat.prop['rockstarId']]

fold_assign = np.zeros(shape=(len(cat),par['nfolds']))

for i in range(par['nfolds']):
    print('\nfold:',i)
    
    fold_assign[fold==i, i] = 2 # Assign test members
    
    log_m = np.log10(cat.prop.loc[(fold!=i).values, 'M200c'])

    bin_edges = np.arange(log_m.min() * 0.9999, (log_m.max() + par['logm_bin'])*1.0001, par['logm_bin'])
    
    n_per_bin = int(par['mbin_frac']*len(log_m)/len(bin_edges))
    
    for j in range(len(bin_edges)):
        bin_ind = log_m.index[ (log_m >= bin_edges[j])&(log_m < bin_edges[j]+par['logm_bin']) ]
        
        if len(bin_ind) <= n_per_bin:
            fold_assign[bin_ind,i] = 1 # Assign train members
            
        else:
            fold_assign[np.random.choice(bin_ind, n_per_bin, replace=False), i] = 1
            
    
    print('in_train:', np.sum(fold_assign[:,i]==1))
    print('in_test:', np.sum(fold_assign[:,i]==2))

# fold_assign marks test/train within each fold. 0 = None, 1 = train, 2 = test.

print('\nRemoving unused data...')
keep = np.sum(fold_assign, axis=1) != 0

fold_assign = fold_assign[keep, :]
cat = cat[keep]



print('\n~~~~~ PREPROCESSING DATA ~~~~~~')

if par['bandwidth'] == 'avg_scott':
    def scott_bandwidth(N, d):
        return N**(-1./(d+4))
        
    bandwidth = scott_bandwidth(cat.prop['Ngal'].mean(), len(par['shape']))
else:
    bandwidth = par['bandwidth']
    
print('bandwidth:', bandwidth)

# Generate input data
print('\nGenerate input data')

mesh = np.mgrid[-cat.par['vcut'] : cat.par['vcut'] : par['shape'][0]*1j,
                0 : cat.par['aperture'] : par['shape'][1]*1j               
                ]

sample = np.vstack([mesh[0].ravel(), mesh[1].ravel()]) # Sample at fixed intervals. Used to sample pdfs


print('Generating ' + str(len(cat)) + ' KDEs...')

progress = list(np.random.randint(0,len(cat),10))

def make_pdf(ind):
    if ind in progress:
        print('marker:', progress.index(ind),'/10' )
        
    memb = np.ndarray(shape=(2,cat.prop.loc[ind, 'Ngal']))

    memb[0,:] = cat.gal[ind]['vlos']
    memb[1,:] = cat.gal[ind]['Rproj']# np.sqrt(cat.gal[i]['xproj']**2 + cat.gal[i]['yproj']**2)
    
    # initialize a gaussian kde from galaxies
    kde = gaussian_kde(memb, bandwidth)

    # sample kde at fixed intervals
    kdeval = np.reshape(kde(sample).T, mesh[0].shape)

    # normalize input
    kdeval /= kdeval.sum()
    
    return kdeval

import time
t0 = time.time()
with mp.Pool(processes=n_proc) as pool:
    pdfs = np.array( pool.map(make_pdf, range(len(cat))) )
print(time.time() - t0)


pdfs = pdfs.astype('float32')

print("KDE's generated.")


print('\nConverting masses, sigv to log scale')

masses = np.log10(cat.prop['M200c']).values
masses = masses.reshape((len(masses),1))

sigv = np.log10(cat.prop['sigv']).values
sigv = sigv.reshape((len(masses),1))





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
    
    'fold_assign':  fold_assign
} 

with open(os.path.join(model_dir, par['model_name'] + '.p'),'wb') as f:
    pickle.dump(save_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

# np.save(os.path.join(model_dir, par['model_name'] + '.npy'), save_dict, allow_pickle=True)

print('Data saved.')







## PLOT CHARACTERISTICS
print('\n~~~~~ PLOTTING MASS DIST ~~~~~')

print('Loading theoretical HMF...')
hmf_M200c = np.loadtxt(os.path.join(par['wdir'], 'data_raw', 'dn_dm_MDPL2_z=0.117_M200c.txt'))

x_hmf_M200c, y_hmf_M200c = hmf_M200c

y_hmf_M200c = x_hmf_M200c*y_hmf_M200c*np.log(10)
x_hmf_M200c = np.log10(x_hmf_M200c)



fold = np.random.randint(0,par['nfolds'])

if fold is None:
    in_train_all = np.sum(fold_assign == 1, axis=1) > 0
    in_test_all = np.sum(fold_assign == 2, axis=1) > 0
else:
    print('Plotting fold #' + str(fold) + '...')
    
    in_train_all = fold_assign[:,fold] == 1
    in_test_all = fold_assign[:,fold] == 2

f, ax = plt.subplots(figsize=(5,5))

ax.semilogy(x_hmf_M200c,y_hmf_M200c, label='theo')


mass_train = masses[in_train_all]
mass_test = masses[in_test_all]

if (len(mass_train) > 0) & (len(mass_test)>0):
    _ = matt.histplot(  mass_train, n=75, log=1, box=True,
                        label='train', ax=ax)
    _ = matt.histplot(  mass_test, n=75, log=1, box=True,
                        label='test', ax=ax)

plt.xlim(mass_test.min(), mass_test.max())

plt.xlabel('$\log(M_{200c})$', fontsize=16)
plt.ylabel('$dn/d\log(M_{200c})$', fontsize=16)

if ~(fold is None):
    plt.title('fold: ' + str(fold), fontsize=20)

plt.legend()
plt.tight_layout()
f.savefig(os.path.join(model_dir, par['model_name'] + '_ydist.pdf'))


print('Figures saved')


print('All finished!')
