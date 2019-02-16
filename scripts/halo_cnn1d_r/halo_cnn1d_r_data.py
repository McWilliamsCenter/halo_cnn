
## IMPORTS

import os
import sys
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
import pickle
import scipy

from collections import OrderedDict
from scipy.stats import gaussian_kde

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
    
    ('data_file'    ,   'Rockstar_UM_z=0.117_contam_med.p'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('shape'        ,   (48,)), # Length of a cluster's ML input array. # of times velocity pdf will be sampled
    
    ('bandwidth'    ,   0.25), # bandwidth used for gaussian kde. Can be scalar, 'scott','silverman', or 'avg_scott'
    
    ('nfolds'       ,   10 ),
    
    ('test_range'   ,   (10**13.9, 10**15.1)),
    
    ('dn_dlogm'		,	10.**-5.2),
    ('dlogm'		,	0.02)

])
# For running
n_proc = 20



## DATA
print('\n~~~~~ LOADING MOCK CATALOG ~~~~~')
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
bin_edges = np.arange(log_m.min() * 0.9999, (log_m.max() + par['dlogm'])*1.0001, par['dlogm'])
n_per_bin = int(par['dn_dlogm']*1000**3*par['dlogm'])

for j in range(len(bin_edges)):
    bin_ind = log_m.index[ (log_m >= bin_edges[j])&(log_m < bin_edges[j]+par['dlogm']) ].values
    
    if len(bin_ind) <= n_per_bin:
        in_train[bin_ind] = True # Assign train members
    else:
        in_train[np.random.choice(bin_ind, n_per_bin, replace=False)] = True

in_test[cat.prop.index[(cat.prop['rotation'] < 3) & 
        (cat.prop['M200c'] > par['test_range'][0]) &
        (cat.prop['M200c'] < par['test_range'][1])].values] = True


print('\n~~~~~ REMOVING UNUSED DATA ~~~~~')
keep = (in_train + in_test) > 0

cat = cat[keep]
in_train = in_train[keep]
in_test = in_test[keep]


print('\n~~~~~ ASSIGNING FOLDS ~~~~~~')
# Use rank-ordering to assign folds evenly for all masses

ids_massSorted = cat.prop[['rockstarId','M200c']].drop_duplicates().sort_values(['M200c','rockstarId'])['rockstarId']

fold_ind = pd.Series(np.arange(len(ids_massSorted)) % par['nfolds'], 
                     index = ids_massSorted)

fold = fold_ind[cat.prop['rockstarId']].values

for i in range(par['nfolds']):
    print('Fold #' + str(i) + ' --> train:' + str(np.sum(in_train[fold!=i])) + ' test:' + str(np.sum(in_test[fold==i])) )


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

sample = np.linspace(-cat.par['vcut'] , cat.par['vcut'], par['shape'][0] + 1)
sample = [np.mean(sample[[i,i+1]]) for i in range(len(sample)-1)] # Sample at fixed intervals. Used to sample pdfs# Sample at fixed intervals. Used to sample pdfs

print('Generating ' + str(len(cat)) + ' KDEs...')

def make_pdf(ind):
    
    # initialize a gaussian kde from galaxy velocities
    kde = gaussian_kde(cat.gal[ind]['vlos'], bandwidth)
    
    # sample kde at fixed intervals
    kdeval = np.reshape(kde(sample).T, par['shape'])
    
    # normalize input
    kdeval /= kdeval.sum()
    
    return kdeval

t0 = time.time()

import tqdm
if n_proc > 1:
    with mp.Pool(processes=n_proc) as pool:
        pdfs = np.array(list( tqdm.tqdm(pool.imap(make_pdf, range(len(cat))), total=len(cat)) ))

else:
    pdfs = np.array(list( tqdm.tqdm(map(make_pdf, range(len(cat)) ), total=len(cat)) ) )
    
print('KDE generation time:',time.time() - t0,'sec')
print('Average generation time:', (time.time()-t0)/len(pdfs),'sec')

print("KDE's generated.")


print('\n~~~~~ BUILDING OUTPUT ARRAY ~~~~~~')
dtype = [
    ('rockstarId','<i8'), ('logmass', '<f4'), ('Ngal','<i8'), ('logsigv','<f4'),
    ('in_train', '?'), ('in_test', '?'), ('fold', '<i4'), ('pdf', '<f4', par['shape'])
]

data = np.ndarray(shape=(len(cat),), dtype=dtype)

data['rockstarId'] = cat.prop['rockstarId'].values
data['logmass'] = np.log10(cat.prop['M200c'].values)
data['Ngal'] = cat.prop['Ngal'].values
data['logsigv'] = np.log10(cat.prop['sigv'].values)
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
    pickle.dump(save_dict, f)

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
    _ = matt.histplot(  mass_train, n=100, log=1, box=True,
                        label='train', ax=ax)
    _ = matt.histplot(  mass_test, n=100, log=1, box=True,
                        label='test', ax=ax)

plt.xlim(mass_train.min(), mass_train.max())

plt.xlabel('$\log(M_{200c})$', fontsize=16)
plt.ylabel('$dn/d\log(M_{200c})$', fontsize=16)

plt.legend()
plt.tight_layout()
f.savefig(os.path.join(model_dir, par['model_name'] + '_ydist.pdf'))


print('Figures saved')

print('All finished!')


