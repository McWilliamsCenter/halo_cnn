
## IMPORTS

import os
import sys
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
import pickle

from collections import OrderedDict
from scipy.stats import gaussian_kde

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
    
    ('data_file'    ,   'Rockstar_UM_z=0.117_contam.p'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('shape'        ,   (48,48)), # Length of a cluster's ML input array. # of times velocity pdf will be sampled 
    
    ('bandwidth'    ,   0.35), # 'avg_scott'), # bandwidth used for gaussian kde. Can be scalar, 'scott','silverman', or 'avg_scott'
    
    ('nfolds'       ,   13),
    ('logm_bin'     ,   0.01),
    ('mbin_frac'    ,   0.008)

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

print('\n~~~~~ ASSIGNING TEST/TRAIN ~~~~~')

in_train = np.array([False]*len(cat))
in_test = np.array([False]*len(cat))

log_m = np.log10(cat.prop['M200c'])
bin_edges = np.arange(log_m.min() * 0.9999, (log_m.max() + par['logm_bin'])*1.0001, par['logm_bin'])
n_per_bin = int(par['mbin_frac']*len(log_m)/len(bin_edges))

for j in range(len(bin_edges)):
    bin_ind = log_m.index[ (log_m >= bin_edges[j])&(log_m < bin_edges[j]+par['logm_bin']) ].values
    
    if len(bin_ind) <= n_per_bin:
        in_train[bin_ind] = True # Assign train members
    else:
        in_train[np.random.choice(bin_ind, n_per_bin, replace=False)] = True

in_test[cat.prop.index[cat.prop['rotation'] < 3].values] = True


print('\n~~~~~ REMOVING UNUSED DATA ~~~~~')
keep = (in_train + in_test) > 0

cat = cat[keep]
in_train = in_train[keep]
in_test = in_test[keep]


print('\n~~~~~ ASSIGNING FOLDS ~~~~~~')

fold_ind = pd.Series(np.random.randint(0, par['nfolds'], len(cat.prop['rockstarId'].unique())), 
                  index = cat.prop['rockstarId'].unique())

fold = fold_ind[cat.prop['rockstarId']].values


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

progress = list(np.random.choice(list(range(0,len(cat))),10))

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


t0 = time.time()
if n_proc > 1:
    with mp.Pool(processes=n_proc) as pool:
        pdfs = np.array( pool.map(make_pdf, range(len(cat))) )
else:
    pdfs = np.array(list(map(make_pdf, range(len(cat)) ) ) )
    
print('KDE generation time:',time.time() - t0,'sec')

print("KDE's generated.")


print('\n~~~~~ BUILDING OUTPUT ARRAY ~~~~~~')
dtype = [
    ('hostid','<i8'), ('logmass', '<f4'), ('in_train', '?'),
    ('in_test', '?'), ('fold', '<i4'), ('pdf', '<f4', par['shape'])
]

data = np.ndarray(shape=(len(cat),), dtype=dtype)

data['hostid'] = cat.prop['rockstarId'].values
data['logmass'] = np.log10(cat.prop['M200c'].values)
data['in_train'] = in_train
data['in_test'] = in_test
data['fold'] = fold
data['pdf'] = pdfs


## SAVE
print('\n~~~~~ SAVE ~~~~~')

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
    
    'data'       :   data
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


f, ax = plt.subplots(figsize=(5,5))

ax.semilogy(x_hmf_M200c,y_hmf_M200c, label='theo')


mass_train = data['logmass'][data['in_train']]
mass_test = data['logmass'][data['in_test']]

if (len(mass_train) > 0) & (len(mass_test)>0):
    _ = matt.histplot(  mass_train, n=75, log=1, box=True,
                        label='train', ax=ax)
    _ = matt.histplot(  mass_test, n=75, log=1, box=True,
                        label='test', ax=ax)

plt.xlim(mass_test.min(), mass_test.max())

plt.xlabel('$\log(M_{200c})$', fontsize=16)
plt.ylabel('$dn/d\log(M_{200c})$', fontsize=16)

plt.legend()
plt.tight_layout()
f.savefig(os.path.join(model_dir, par['model_name'] + '_ydist.pdf'))


print('Figures saved')

print('All finished!')
