

## IMPORTS

import sys
import os
import numpy as np
import pickle
import pandas as pd

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
    ('model_name'   ,   'halo_cnn2d_rP'),
    
    ('pure_cat'     ,   'data_mocks/Rockstar_UM_z=0.117_pure.p'),
    ('contam_cat'   ,   'data_mocks/Rockstar_UM_z=0.117_contam_med.p'),
    
    ('out_folder'   ,   'data_processed'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('use_proc'     ,   False),

    ('train_range'  ,   (10**14.5, 10**15.3)),
    ('test_range'   ,   (10**13.5, 10**15.3)),

    ('rel_mass'     ,   10**15),
    
    ('dn_dlogm'     ,   10.**-5.2),
    ('dlogm'        ,   0.02)
    
])

dtype = [
        ('rockstarId','<i8'), ('logmass','<f4'), ('Ngal','<i8'), ('logsigv','<f4'),
        ('in_train', '?'), ('in_test', '?')
    ]


## DATA

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
    """
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
    """

    # remove unused data
    keep = (in_train + in_test) > 0

    cat = cat[keep]
    in_train = in_train[keep]
    in_test = in_test[keep]

    
    # building recarray
    
    data = np.ndarray(shape=(len(cat),), dtype=dtype)

    data['rockstarId'] = cat.prop['rockstarId'].values
    data['logmass'] = np.log10(cat.prop['M200c'].values)
    data['Ngal'] = cat.prop['Ngal'].values
    data['logsigv'] = np.log10(cat.prop['sigv'].values)
    data['in_train'] = in_train
    data['in_test'] = in_test
    
    return data


print('\n~~~~~ LOADING PURE DATA ~~~~~')
data_pure = load_from_cat(os.path.join(par['wdir'], par['pure_cat']))

if par['use_proc']:
    print('\n~~~~~ LOADING PROCESSED CONTAM DATA ~~~~~')

    data_path = os.path.join(par['wdir'], 'data_processed', par['model_name'])
    with open(os.path.join(data_path, par['model_name'] + '.p'), 'rb') as f:
        data_dict = pickle.load(f)
    
    data_contam = data_dict['data'][[i[0] for i in dtype]]
    del data_dict
else:
    print('\n~~~~~ LOADING CONTAM DATA ~~~~~')
    data_contam = load_from_cat(os.path.join(par['wdir'], par['contam_cat']))


## REGRESSION
def run_regression(dat):
    
    preds = np.ndarray(shape=(len(dat),), 
                       dtype=[('logmass','<f4'), ('logmass_pred','<f4'), 
                              ('in_train','?'),  ('in_test','?')])

    preds['logmass'] = dat['logmass']
    preds['in_train'] = dat['in_train'] & (dat['logmass'] > np.log10(par['train_range'][0])) & \
                                          (dat['logmass'] < np.log10(par['train_range'][1]))
    preds['in_test'] = dat['in_test']
    
    regr_models = np.ndarray(shape=(1,), 
                             dtype=[('coef','<f4'), ('intercept','<f4'), 
                                    ('R^2','<f4'), ('scatter','<f4') ])

        
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(dat['logmass'][dat['in_train']].reshape(-1, 1) - np.log10(par['rel_mass']), 
             dat['logsigv'][dat['in_train']])
             
    regr_models['coef'] = regr.coef_
    regr_models['intercept'] = regr.intercept_
    regr_models['R^2'] = regr.score(dat['logmass'][dat['in_train']].reshape(-1, 1), 
                                       dat['logsigv'][dat['in_train']])
    regr_models['scatter'] = np.std(regr.predict(dat['logmass'][dat['in_train']].reshape(-1, 1) - \
                                                 np.log10(par['rel_mass'])) - dat['logsigv'][dat['in_train']])
    
    print('coef:', regr_models['coef'])
    print('intercept:', regr_models['intercept'])
    print('R^2:', regr_models['R^2'])
    print('scatter:', regr_models['scatter'])

    print('sigma_v,' + str(int(np.log10(par['rel_mass']))) + '=',10**regr_models['intercept'][0], 'km/s')
    print('alpha =', regr_models['coef'][0])
    
    preds['logmass_pred'][(dat['in_test'])] = \
        (dat['logsigv'][dat['in_test']] - regr.intercept_)/regr.coef_ + np.log10(par['rel_mass'])
        
    return regr_models, preds
    
print('\n~~~~~ PURE REGRESSION ~~~~~~')
regr_pure, pred_pure = run_regression(data_pure)

print('\n~~~~~ CONTAM REGRESSION ~~~~~~')
regr_contam, pred_contam = run_regression(data_contam)


print('\n~~~~~ FILE SPECIFICATION ~~~~~')
## FILE SPECIFICATION

save_dir = os.path.join(par['wdir'], 'saved_models', par['model_name'])

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Assign most recent model number
log = os.listdir(save_dir)

log = [int(i.split('_')[-1]) for i in log if (i[:-len(i.split('_')[-1])-1]== par['model_name'])]

par['model_num'] = 0 if len(log)==0 else max(log)
model_name_save = par['model_name'] + '_' + str(par['model_num'])
model_dir = os.path.join(save_dir, model_name_save)


print('\n~~~~~ BUILDING OUTPUT ~~~~~')
save_dict = {
    'pure' : {'model': regr_pure, 'pred':pred_pure},
    'contam' : {'model': regr_contam, 'pred':pred_contam}
}

print('~~~~~ SAVING DATA ~~~~~~')
with open(os.path.join(model_dir, model_name_save + '_regr.p'),'wb') as f:
    pickle.dump(save_dict, f)




print('~~~~~ PLOTTING REGRESSION ~~~~~~')

f = plt.figure(figsize=[4,6])
gs = mpl.gridspec.GridSpec(2,1,height_ratios=[1,1], hspace=0)

ax1 = f.add_subplot(gs[0,0])

matt.binnedplot( data_pure['logmass'][data_pure['in_train']], 
                 data_pure['logsigv'][data_pure['in_train']],
                 n=50,
                 percentiles = [35,47.5],
                 ax=ax1,
                 label='pure train',
                 names=False,
                 c='b',
                 log=0
                )
matt.binnedplot( data_contam['logmass'][data_contam['in_train']], 
                 data_contam['logsigv'][data_contam['in_train']],
                 n=50,
                 percentiles = [35,47.5],
                 ax=ax1,
                 label='contam train',
                 names=False,
                 c='r',
                 log=0
                )

one_to_one = np.arange(11)*(data_contam['logmass'].max() - data_contam['logmass'].min())/10. + data_contam['logmass'].min()

fit_y = (one_to_one - np.log10(par['rel_mass'])) * regr_pure['coef'] + regr_pure['intercept']
ax1.plot(one_to_one,fit_y,'b',label='pure fit')
fit_y = (one_to_one - np.log10(par['rel_mass'])) * regr_contam['coef'] + regr_contam['intercept']
ax1.plot(one_to_one,fit_y,'r',label='contam fit')

ax1.axvline(x=np.log10(par['train_range'][0]), linestyle='--', color='k')


ax1.set_xlim(xmin=data_contam['logmass'].min(), xmax=data_contam['logmass'].max())
ax1.set_xticks([])

ax1.set_ylabel('$\log[\sigma_v]$', fontsize=14)
ax1.legend(fontsize=8,loc=4)


ax2 = f.add_subplot(gs[1,0],sharex=ax1)
                
ax2.plot(one_to_one,one_to_one,'k',linestyle='dashed')

matt.binnedplot( pred_pure['logmass'][data_pure['in_test']],
                 pred_pure['logmass_pred'][data_pure['in_test']],
                 n=50,
                 percentiles = [35],
                 ax=ax2,
                 label='pure_test',
                 names=False,
                 c='b',
                 log=0
                )
matt.binnedplot( pred_contam['logmass'][data_contam['in_test']],
                 pred_contam['logmass_pred'][data_contam['in_test']],
                 n=50,
                 percentiles = [35],
                 ax=ax2,
                 label='contam_test',
                 names=False,
                 c='r',
                 log=0
                )
                
ax2.set_xlim(xmin=data_contam['logmass'].min(), xmax=data_contam['logmass'].max())
ax2.set_ylim(ymin=data_contam['logmass'].min(), ymax=data_contam['logmass'].max())

ax2.set_xlabel('$\log[M$]', fontsize=14)
ax2.set_ylabel('$\log[M_{pred}$]', fontsize=14)
ax2.legend(fontsize=8, loc=4)

plt.tight_layout()
f.savefig(os.path.join(model_dir, model_name_save+ '_regr.pdf'))


