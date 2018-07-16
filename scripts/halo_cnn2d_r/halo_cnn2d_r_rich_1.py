
## IMPORTS

import sys
import os
import numpy as np
import multiprocessing as mp
import pickle
import time

from scipy.stats import gaussian_kde
from collections import OrderedDict

## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Catalog


## PARAMETERS
par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_cnn2d_r'),

    ('cat'          ,   'data_mocks/Rockstar_UM_z=0.117_contam.p'),
    ('Nkde'         ,   1000),
    
    ('mdim'         ,   1),
    ('ndim'         ,   1)
])
n_proc=1

print('\n~~~~~ LOADING CATALOG ~~~~~')
cat = Catalog().load(os.path.join(par['wdir'], par['cat']))


print('\n~~~~~ LOADING PROCESSED DATA ~~~~~')
data_path = os.path.join(par['wdir'], 'data_processed', par['model_name'])
with open(os.path.join(data_path, par['model_name'] + '.p'), 'rb') as f:
    dict_proc = pickle.load(f)
par_proc = dict_proc['params']
data_proc = dict_proc['data']

print('\n~~~~~ REDUCING DATASET ~~~~~')
data_proc = data_proc[data_proc['in_test']]
data_proc = data_proc[data_proc['logmass']>15]
uniq_ids = set(data_proc['hostid'])   
cat = cat[[i in uniq_ids for i in cat.prop['rockstarId'].values ]]
    
print('\n~~~~~ CHOOSING CLUSTERS ~~~~~')

par['logmass_min'] = data_proc['logmass'].min()
par['logmass_max'] =  data_proc['logmass'].max()

m_bins = np.linspace(par['logmass_min'], par['logmass_max'], par['mdim']+1)

cluster_ids = []

for i in range(par['mdim']):
    bin_data = data_proc[(data_proc['logmass'] > m_bins[i]) & (data_proc['logmass'] <= m_bins[i+1])]
                                 
    log_N = np.log10(bin_data['Ngal'])
                                       
    n_bins = np.linspace(log_N.min(), log_N.max(), par['ndim']+1)
    
    for j in range(par['ndim']):
        cluster_ids.append(np.random.choice(bin_data['hostid'][(log_N > n_bins[j]) & (log_N <= n_bins[j+1])]))

print(cluster_ids)

print('\n~~~~~ GENERATING KDEs ~~~~~')
mesh = np.mgrid[-cat.par['vcut'] : cat.par['vcut'] : par_proc['shape'][0]*1j,
                0 : cat.par['aperture'] : par_proc['shape'][1]*1j               
                ]

sample = np.vstack([mesh[0].ravel(), mesh[1].ravel()]) # Sample at fixed intervals. Used to sample pdfs

meta_dtype = [('log_mass', '<f4'), ('Ngal','<i8'), ('fold','<i4'), ('true_frac','<f4')]
main_dtype = [('pdfs','<f4', (par['Nkde'], *par_proc['shape'] ) ), ('richs', '<f4', par['Nkde'])]

def sample_rich(rId):
    
    i_proc = np.random.choice(np.where(data_proc['hostid']==rId)[0])
    i_cat = np.random.choice(np.where( (cat.prop['rockstarId'].values == rId) & (cat.prop['Ngal'].values == data_proc['Ngal'][i_proc]))[0])
                                        
    
    meta_data = np.ndarray(shape=(1,), dtype=meta_dtype)
    
    meta_data['log_mass'] = data_proc['logmass'][i_proc]
    meta_data['Ngal'] = data_proc['Ngal'][i_proc]
    meta_data['fold'] = data_proc['fold'][i_proc]
    meta_data['true_frac'] = np.sum(cat.gal[i_cat]['true_memb'])/data_proc['Ngal'][i_proc]
    
    main_data = np.ndarray(shape=(1,), dtype=main_dtype)
    
    Ngal = meta_data['Ngal'][0]
    n_per_bin = int((par['Nkde']-1)/ (Ngal-2))
    
    i = 0

    for n in np.arange(2, Ngal+1):

        k = n_per_bin if n!=Ngal else ((par['Nkde']-1) % (Ngal-2))+1

        for j in range(k):
            ind = np.random.choice(range(Ngal), size=n, replace=False)
            
            memb = np.ndarray(shape=(2, n))

            memb[0,:] = cat.gal[i_cat]['vlos'][ind]
            memb[1,:] = cat.gal[i_cat]['Rproj'][ind]
            
            # initialize a gaussian kde from galaxies
            kde = gaussian_kde(memb, par_proc['bandwidth'])
                
            # sample kde at fixed intervals
            kdeval = np.reshape(kde(sample).T, mesh[0].shape)
            
            # check for nans
            if np.isnan(kdeval).any():
                kdeval = 0
            else:
                # normalize input
                s = kdeval.sum()
                if s > 0:
                    kdeval = kdeval/(kdeval.sum())
                else:
                    kdeval = 0
            
            main_data[0]['pdfs'][i] = kdeval
            main_data[0]['richs'][i] = n
            
            i+=1
    
    return meta_data, main_data


t0 = time.time()
if n_proc > 1:
    with mp.Pool(processes=n_proc) as pool:
        dat = np.array( pool.map(sample_rich, cluster_ids) )
else:
    dat = np.array(list(map(sample_rich, cluster_ids) ) )
    
print('KDE generation time:',time.time() - t0,'sec')

print("KDE's generated.")

for i in range(len(dat)):
    if i==0:
        meta_data = dat[i][0]
        main_data = dat[i][1]
    else:
        meta_data = np.append(meta_data, dat[i][0])
        main_data = np.append(main_data, dat[i][1])

print(meta_data)

save_dict = OrderedDict([
    ('meta'         ,   meta_data),
    ('main'         ,   main_data),
    ('logmass_min'  ,   par['logmass_min']),
    ('logmass_max'  ,   par['logmass_max']),
    ('shape'        ,   (par['mdim'], par['ndim']) ) 
])


with open(os.path.join(data_path, par['model_name'] + '_rich.p'),'wb') as f:
    pickle.dump(save_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

print('Data saved.')





