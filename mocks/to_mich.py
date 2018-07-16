
""" 
Converts Catalog dataset to a .npy file in Michelle's original format.
"""


import sys
import os

import pickle
import numpy as np
import pandas as pd

from collections import OrderedDict, defaultdict

from tools.catalog import Catalog


## ~~~~~~ PARAMETERS ~~~~~~
par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn'),
    ('in_folder'    ,   'data_mocks'),
    ('out_folder'   ,   'data_mocks'),
    
    ('in_file'      ,   'Rockstar_UM_z=0.117_contam_rot10.p'),
    ('out_file'     ,   'Rockstar_UM_z=0.117_contam_rot10_mich.npy'),
    
    ('subsample'    ,   1.0),
    
    ('nfolds'        ,   10),
    ('mbin_frac'    ,   1/40.),
    ('logm_bin'     ,   0.01)

    ])

## ~~~~~ LOADING DATA ~~~~~
print('\n~~~~~ LOADING DATA ~~~~~')

in_path = os.path.join(par['wdir'], par['in_folder'], par['in_file'])

cat = Catalog().load(in_path)



if par['subsample'] < 1:
    
    print('\n~~~~~ SUBSAMPLING ~~~~~')
    print('Original catalog length:', len(cat))
    
    ind = np.random.choice(range(len(cat)), 
                           int(par['subsample']*len(cat)),
                           replace=False
                          )
    cat = cat[ind]
    
    print('New catalog length:', len(cat))
    


print('\n~~~~~ INITIALIZING OUTPUT ~~~~~')
dtype = [('Mtot', '<f4'), ('hostid', '<i8'),  
         ('rotation', '<i4'), ('fold', '<i4'), ('Ngal', '<i4'), 
         ('vlos', '<f4', (1200,)), ('sigmav', '<f4'), ('Rproj', '<f4', (1200,)), 
         ('xyproj', '<f4', (1200, 2)), ('truememb', '?', (1200,)), 
         ('intest', '<i4'), ('intrain', '<i4'), ('redshift', '<f4'), ('Rs', '<f4')]

dat_mich = np.ndarray(shape=(len(cat),), dtype=dtype)



print('\n~~~~~ ASSIGNING TRAIN, TEST ~~~~~')

log_m = np.log10(cat.prop['M200c'])

bin_edges = np.arange(log_m.min() * 0.9999, (log_m.max() + par['logm_bin'])*1.0001, par['logm_bin'])
    
n_per_bin = int(par['mbin_frac']*len(log_m)/len(bin_edges))

for j in range(len(bin_edges)):
    bin_ind = log_m.index[ (log_m >= bin_edges[j])&(log_m < bin_edges[j]+par['logm_bin']) ].values
    
    if len(bin_ind) <= n_per_bin:
        dat_mich['intrain'][bin_ind] = 1 # Assign train members
        
    else:
        dat_mich['intrain'][np.random.choice(bin_ind, n_per_bin, replace=False)] = 1

dat_mich['intest'][cat.prop.index[cat.prop['rotation'] < 3].values] = 1



print('\n~~~~~ REMOVING UNUSED DATA ~~~~~')
keep = (dat_mich['intrain'] + dat_mich['intest']) > 0

dat_mich = dat_mich[keep]
cat = cat[keep]



print('\n~~~~~ ASSIGNING FOLDS ~~~~~~')
fold_ind = pd.Series(np.random.randint(0, par['nfolds'], len(cat.prop['rockstarId'].unique())), 
                  index = cat.prop['rockstarId'].unique())

dat_mich['fold'] = fold_ind[cat.prop['rockstarId']].values



print('\n~~~~~ COPYING CLUSTER INFO ~~~~~')
dat_mich['Mtot'] = cat.prop['M200c'].values
dat_mich['hostid'] = cat.prop['rockstarId'].values

dat_mich['rotation'] = cat.prop['rotation'].values
dat_mich['Ngal'] = cat.prop['Ngal'].values
dat_mich['Rs'] = cat.prop['Rs'].values
dat_mich['sigmav'] = cat.prop['sigv'].values

dat_mich['redshift'].fill(cat.par['z'])



print('\n~~~~~ COPYING GALAXY INFO ~~~~~')
for i in range(len(cat)):
    if i%int(len(cat)/10)==0: print(int(i/int(len(cat)/10)), '/10')
    
    dat_mich['vlos'][i][0:cat.prop.loc[i,'Ngal']] = cat.gal[i]['vlos']
    dat_mich['Rproj'][i][0:cat.prop.loc[i,'Ngal']] = cat.gal[i]['Rproj']
    dat_mich['xyproj'][i][0:cat.prop.loc[i,'Ngal'], 0] = cat.gal[i]['xproj']
    dat_mich['xyproj'][i][0:cat.prop.loc[i,'Ngal'], 1] = cat.gal[i]['yproj']
    dat_mich['truememb'][i][0:cat.prop.loc[i,'Ngal']] = cat.gal[i]['true_memb']

print('\n~~~~~ SAVING ~~~~~~')

out_path = os.path.join(par['wdir'], par['out_folder'], par['out_file'])
print('Saving to ' + out_path + ' ...')

np.save(out_path, dat_mich)

print('All done!')
