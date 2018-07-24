

## IMPORTS

import sys
import os
import numpy as np
import pickle

from sklearn import linear_model
from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Catalog

## PLOT PARAMETERS
par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_cnn2d_r'),
    
    ('pure_cat'     ,   'data_mocks/Rockstar_UM_z=0.117_pure.p'),
    ('contam_cat'   ,   'data_mocks/Rockstar_UM_z=0.117_contam.p'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('use_proc'     ,   True),
    
    ('nfolds'       ,   15),
    ('logm_bin'     ,   0.01),
    ('mbin_frac'    ,   0.008),
    
])

def load_from_cat(filename):
    # load
    cat = Catalog().load(filename)
    
    # subsample
    print('Data length: ' + str(len(cat)))
    if par['subsample'] < 1:
        ind = np.random.choice(range(len(cat)), 
                               int(par['subsample']*len(cat)),
                               replace=False
                              )
        cat = cat[ind]

        print('Subsampled data length: ' + str(len(cat)))
    
    
    # assign train/test
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


    # remove unused data
    keep = (in_train + in_test) > 0

    cat = cat[keep]
    in_train = in_train[keep]
    in_test = in_test[keep]


    # assign folds
    fold_ind = pd.Series(np.random.randint(0, par['nfolds'], len(cat.prop['rockstarId'].unique())), 
                      index = cat.prop['rockstarId'].unique())

    fold = fold_ind[cat.prop['rockstarId']].values
    
    # building recarray
    dtype = [
        ('rockstarId','<i8'), ('logmass','<f4'), ('Ngal','<i8'), ('logsigv','<f4'),
        ('in_train', '?'), ('in_test', '?'), ('fold', '<i4')
    ]
    
    data = np.ndarray(shape=(len(cat),), dtype=dtype)

    data['rockstarId'] = cat.prop['rockstarId'].values
    data['logmass'] = np.log10(cat.prop['M200c'].values)
    data['Ngal'] = cat.prop['Ngal'].values
    data['logsigv'] = np.log10(cat.prop['sigv'].values)
    data['in_train'] = in_train
    data['in_test'] = in_test
    data['fold'] = fold
    
    return data


## DATA
print('\n~~~~~ LOADING PURE DATA ~~~~~')


