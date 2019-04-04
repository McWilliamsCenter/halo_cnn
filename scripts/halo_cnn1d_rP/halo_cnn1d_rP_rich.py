
## IMPORTS

import sys
import os
import numpy as np
import multiprocessing as mp
import pickle
import time
import keras
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from collections import OrderedDict

## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Catalog


## PARAMETERS
par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_cnn1d_r'),

    ('cat'          ,   'data_mocks/Rockstar_UM_z=0.117_contam_med.p'),
    
    ('subsample'    ,   10.**-1),
    ('n_per_cluster',   5),
    
    ('err_type'     ,   'pred'), #pred
    
    ('plot_indiv'   ,   False),
    ('plot_scale'   ,   4),
    ('plot_width'   ,   4),
    
    ('M_bins'       ,   4),
    ('N_bins'       ,   4),

    ('rel_mass'     ,   10**15)
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

if par['subsample'] < 1:
    print('\n~~~~~ SUBSAMPLING ~~~~~')
    
    data_proc = np.random.choice(data_proc, int(par['subsample']*len(data_proc)), replace=False)


print('\n~~~~~ REDUCING DATASET ~~~~~')
data_proc = data_proc[data_proc['in_test']]
uniq_ids = set(data_proc['rockstarId'])
cat =  cat[[i in uniq_ids for i in cat.prop['rockstarId'].values ]]

print('Data length:', len(data_proc))


print('\n~~~~~ FILE SPECIFICATION ~~~~~')
## FILE SPECIFICATION

save_dir = os.path.join(par['wdir'], 'saved_models', par['model_name'])

# Assign most recent model number
log = os.listdir(save_dir)
log = [int(i.split('_')[-1]) for i in log if (i[:-len(i.split('_')[-1])-1]== par['model_name'])]

par['model_num'] = 0 if len(log)==0 else max(log)

model_name_save = par['model_name'] + '_' + str(par['model_num'])
model_fold_dir = os.path.join(save_dir, model_name_save,'models')

print('Model:', model_name_save)


print('\n~~~~~ LOADING REGRESSION MODELS ~~~~~')
with open(os.path.join(save_dir, model_name_save, model_name_save + '_regr.p'),'rb') as f:
    regr_data = pickle.load(f)
regr_models = regr_data['contam']['model']

print('\n~~~~~ LOADING KERAS MODELS ~~~~~~')
keras_models = [keras.models.load_model(os.path.join(model_fold_dir, 'fold_' + str(i) + '.h5'))
                for i in range(data_proc['fold'].max()+1)
               ]

print('\n~~~~~ ASSIGNING MULTIPLICITY ~~~~~')
rId_to_icat = pd.Series(cat.prop.index.values, index = cat.prop['rockstarId'].values)

dtype = [('gen','<i4'),('mbin','<i4'),('nbin','<i4')]
multi = np.ndarray(shape=(len(data_proc),), dtype=dtype)

multi['gen'] = par['n_per_cluster']
multi['mbin'] = 1
multi['nbin'] = 1

# normalize mbin
bins = np.linspace(data_proc['logmass'].min(), data_proc['logmass'].max(), par['M_bins'] + 1)
bin_inds = [np.where((data_proc['logmass'] >= bins[i]) & (data_proc['logmass'] < bins[i+1]))[0] for i in range(len(bins)-1)]

goal_len = len(max(bin_inds, key=lambda x: len(x)))

for i in range(len(bin_inds)):
    to_add = np.random.choice(bin_inds[i], goal_len - len(bin_inds[i]), replace=True)
    for j in to_add: 
        multi[j]['mbin'] += 1
        
# normalize nbin
bins = np.linspace(np.log10(data_proc['Ngal'].min()), np.log10(data_proc['Ngal'].max()), par['N_bins'] + 1)
bin_inds = [np.where((np.log10(data_proc['Ngal']) >= bins[i]) & (np.log10(data_proc['Ngal']) < bins[i+1]))[0] for i in range(len(bins)-1)]

goal_len = len(max(bin_inds, key=lambda x: len(x)))

for i in range(len(bin_inds)):
    to_add = np.random.choice(bin_inds[i], goal_len - len(bin_inds[i]), replace=True)
    for j in to_add: 
        multi[j]['nbin'] += 1

print('\n~~~~~ GENERATING KDEs, MAKING KERAS/SIGV PREDICTIONS ~~~~~')

sample = np.linspace(-cat.par['vcut'] , cat.par['vcut'], par_proc['shape'][0] + 1)
sample = [np.mean(sample[[i,i+1]]) for i in range(len(sample)-1)] # Sample at fixed intervals. Used to sample pdfs

dtype = [('rockstarId', '<i8'), ('logmass', '<f4'), ('N', '<i4'), ('N1.0', '<i4'),
         ('rich_frac', '<f4'), ('true_frac','<f4'),
         ('logmass_keras','<f4'), ('logmass_keras1.0', '<f4'), ('err_keras', '<f4'),
         ('logmass_sigv','<f4'), ('logmass_sigv1.0', '<f4'), ('err_sigv', '<f4'),
         ('gen','?'),('mbin','?'),('nbin','?'),     
        ]
        
logmass_bounds = (data_proc['logmass'].min(), data_proc['logmass'].max())

def sample_rich(i_proc):
    if i_proc%100==0: print(i_proc)
    
    rId = data_proc['rockstarId'][i_proc]
    fold = data_proc['fold'][i_proc]

    i_cat = np.random.choice(np.where( (cat.prop['rockstarId'].values == rId) & (cat.prop['Ngal'].values == data_proc['Ngal'][i_proc]))[0])
    
    out = np.ndarray(shape=(max(multi[i_proc]) + 1,), dtype=dtype)
    
    out['rockstarId'] = rId
    out['logmass'] = data_proc['logmass'][i_proc]
    out['N1.0'] = data_proc['Ngal'][i_proc]

    out[['gen','mbin','nbin']] = 0
    out['gen'][1: 1+multi[i_proc]['gen']] = 1
    out['mbin'][1: 1+multi[i_proc]['mbin']] = 1
    out['nbin'][1: 1+multi[i_proc]['nbin']] = 1
    
    for i in range(max(multi[i_proc]) + 1):
        if i==0: 
            rich = data_proc['Ngal'][i_proc]
        else:
            rich = int(np.random.rand() * (data_proc['Ngal'][i_proc]-3) + 3)
    
    
        ind = np.random.choice(range(data_proc['Ngal'][i_proc]), 
                                size = rich, 
                                replace=False)
        memb = np.ndarray(shape=(1, rich))
        memb[0,:] = cat.gal[i_cat]['vlos'][ind]
        
        # initialize a gaussian kde from galaxies
        kde = gaussian_kde(memb, par_proc['bandwidth'])

        # sample kde at fixed intervals
        kdeval = np.reshape(kde(sample).T, [1] + list(par_proc['shape']) + [1] )
        kdeval = kdeval/(kdeval.sum())
        
        out[i]['N'] = rich
        out[i]['rich_frac'] = rich/data_proc['Ngal'][i_proc]
        out[i]['true_frac'] = np.sum(cat.gal[i_cat]['true_memb'])/data_proc['Ngal'][i_proc]
         
        out[i]['logmass_keras'] = (logmass_bounds[1] - logmass_bounds[0]) * keras_models[fold].predict(kdeval) + logmass_bounds[0]
              
        out[i]['logmass_sigv'] = (np.log10(memb[0,:].std()) - regr_models['intercept'])/regr_models['coef']+ np.log10(par['rel_mass'])

        if i==0:
            out['logmass_keras1.0'] = out[i]['logmass_keras']
            out['logmass_sigv1.0'] = out[i]['logmass_sigv']
    
    if par['err_type'] == 'true':
        out['err_keras'] = 10**(out['logmass_keras'] - out['logmass']) - 1
        out['err_sigv'] = 10**(out['logmass_sigv'] - out['logmass']) - 1
        
    elif par['err_type'] == 'pred':
        out['err_keras'] = 10**(out['logmass_keras'] - out['logmass_keras1.0']) - 1
        out['err_sigv'] = 10**(out['logmass_sigv'] - out['logmass_sigv1.0']) - 1
    
    else:
        raise Exception('Unknown err_type: ' + str(par['err_type']))
    
    return out


t0 = time.time()
if n_proc > 1:
    with mp.Pool(processes=n_proc) as pool:
        data_rich = pool.map(sample_rich, range(len(data_proc)))
else:
    data_rich = list(map(sample_rich, range(len(data_proc))) )

print('KDE generation time:',time.time() - t0,'sec')

print("KDE's generated.")


if par['plot_indiv']:
    print('\n~~~~~ PLOTTING INDIVIDUAL RICHNESS DEPENDANCE ~~~~~~')

    data_rich.sort(key = lambda x: x[0]['logmass'])
    
    shape = (int(len(data_rich)/par['plot_width']) + 1, par['plot_width'])
    

    f = plt.figure(figsize=[shape[1]*par['plot_scale'],shape[0]*par['plot_scale']])
    gs = mpl.gridspec.GridSpec(*shape)
    
    for i in range(len(data_rich)):
        ax = f.add_subplot(gs[i%shape[0],int(i/shape[0])])

        matt.binnedplot(data_rich[i]['rich_frac'], data_rich[i]['err_keras'],
                        n=20, percentiles = [34,47.5],
                        median = True, ax=ax, log=0, c='b', label='keras'
                       )
        matt.binnedplot(data_rich[i]['rich_frac'], data_rich[i]['err_sigv'],
                        n=20, percentiles = [34,47.5],
                        median = True, ax=ax, log=0, c='r', label='$M(\sigma)$'
                       )

        ax.axhline(y=data_rich[i]['err_keras'][0], linestyle='dashed', c='k')
        ax.set_xlim(0,1)
        ax.set_ylim(-1, 2*np.abs(data_rich[i]['err_keras'][0]+1) - 1)
        ax.set_xlabel('Richness Fraction')
        ax.set_title('$\log(M_{200c})=$ ' + str(data_rich[i]['logmass'][0])[:5] + 
                     ' ; \n $N=$' + str(data_rich[i]['N'][0]) + ' ; $f_t=$' + 
                     str(data_rich[i]['true_frac'][0])[:5])

        if par['err_type']=='true':
            ax.set_ylabel('$\epsilon$')
        elif par['err_type'] == 'pred':
            ax.set_ylabel('$\epsilon_{1.0}$')
            
    plt.tight_layout()
    f.savefig(os.path.join(save_dir, model_name_save, model_name_save + '_rich.pdf')) 

print('\n~~~~~ SAVING OUTPUT ~~~~~')
data_rich = np.hstack(data_rich)

np.save(os.path.join(save_dir, model_name_save, model_name_save + '_rich.npy'), data_rich)

print('\n~~~~~ PLOTTING MARGINALIZED RICHNESS DEPENDANCE ~~~~~~')

f,ax = plt.subplots(figsize=(4,4))

matt.binnedplot(data_rich['rich_frac'][data_rich['gen']==1], data_rich['err_sigv'][data_rich['gen']==1],
                n=20, percentiles = [34,47.5],
                median = True, ax=ax, log=0, c='r', label='$M(\sigma)$', names=False
               )
matt.binnedplot(data_rich['rich_frac'][data_rich['gen']==1], data_rich['err_keras'][data_rich['gen']==1],
                n=20, percentiles = [34,47.5],
                median = True, ax=ax, log=0, c='b', label='cnn', names=False
               )

ax.axhline(y = 0, linestyle='dashed', c = 'k')
ax.set_xlim(0,1)
ax.set_ylim(-1, 1)
ax.set_xlabel('Richness Fraction', fontsize=12)

if par['err_type']=='true':
    ax.set_ylabel('$\epsilon$', fontsize=12)
elif par['err_type'] == 'pred':
    ax.set_ylabel('$\epsilon_{1.0}$', fontsize=12)

plt.legend(fontsize=12)
plt.tight_layout()
f.savefig(os.path.join(save_dir, model_name_save, model_name_save + '_richmargin.pdf'))


print('\n~~~~~ PLOTTING MARGINALIZED RICHNESS DEPENDANCE AT BINNED MASSES ~~~~~~')

f,ax = plt.subplots(figsize=(4,4))

bins = np.linspace(data_rich['logmass'].min(), data_rich['logmass'].max(), par['M_bins'] + 1)
colr = ['b','r','g','c','m','y']

for i in range(par['M_bins']):
    data_bin = data_rich[(data_rich['logmass'] >= bins[i]) & (data_rich['logmass'] < bins[i+1]) &
                         (data_rich['mbin'])]
    
    if len(data_bin)>0:
        matt.binnedplot(data_bin['rich_frac'], data_bin['err_keras'],
                        n=20, percentiles = [34],
                        median = True, ax=ax, log=0, c= colr[i%len(colr)], names=False,
                        label='[' + str(bins[i])[0:5] + ', ' + str(bins[i+1])[0:5] + ']'
                       )

ax.axhline(y = 0, linestyle='dashed', c = 'k')
ax.set_xlim(0,1)
ax.set_ylim(-1, 1)
ax.set_xlabel('Richness Fraction', fontsize=12)

if par['err_type']=='true':
    ax.set_ylabel('$\epsilon$', fontsize=12)
elif par['err_type'] == 'pred':
    ax.set_ylabel('$\epsilon_{1.0}$', fontsize=12)

plt.legend(fontsize=9, ncol=2, title='$\log[M_{200c}\ ($M$_\odot h^{-1})]$',
           handletextpad=0.25, columnspacing=0.75, handlelength=1.3, loc=9
          )
plt.tight_layout()
# plt.title('Binned logmass', fontsize=24)
f.savefig(os.path.join(save_dir, model_name_save, model_name_save + '_richmargin_mbin.pdf'))

print('\n~~~~~ PLOTTING MARGINALIZED RICHNESS DEPENDANCE AT BINNED RICHNESS ~~~~~~')

f,ax = plt.subplots(figsize=(4,4))

bins = np.linspace(np.log10(data_rich['N1.0'].min()), np.log10(data_rich['N1.0'].max()), par['N_bins'] + 1)
colr = ['b','r','g','c','m','y']

for i in range(par['N_bins']):
    data_bin = data_rich[(np.log10(data_rich['N1.0']) >= bins[i]) & 
                         (np.log10(data_rich['N1.0']) < bins[i+1]) & (data_rich['nbin'])]
    
    if len(data_bin)>0:
        matt.binnedplot(data_bin['rich_frac'], data_bin['err_keras'],
                        n=20, percentiles = [34],
                        median = True, ax=ax, log=0, c= colr[i%len(colr)], names=False,
                        label='[' + str(bins[i])[0:5] + ', ' + str(bins[i+1])[0:5] + ']'
                       )

ax.axhline(y = 0, linestyle='dashed', c = 'k')
ax.set_xlim(0,1)
ax.set_ylim(-1, 1)
ax.set_xlabel('Richness Fraction', fontsize=12)

if par['err_type']=='true':
    ax.set_ylabel('$\epsilon$', fontsize=12)
elif par['err_type'] == 'pred':
    ax.set_ylabel('$\epsilon_{1.0}$', fontsize=12)

plt.legend(fontsize=9, ncol=2, title='$\log[N_{1.0}]$', loc=9,
           handletextpad=0.25, columnspacing=0.75, handlelength=1.3
          )
plt.tight_layout()
# plt.title('Binned richness', fontsize=24)
f.savefig(os.path.join(save_dir, model_name_save, model_name_save + '_richmargin_nbin.pdf'))

print('All done!')
