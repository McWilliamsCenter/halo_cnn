

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
    ('model_name'   ,   'halo_cnn2d_r'),
    
    ('pure_cat'     ,   'data_mocks/Rockstar_UM_z=0.117_pure.p'),
    ('contam_cat'   ,   'data_mocks/Rockstar_UM_z=0.117_contam_med.p'),
    
    ('out_folder'   ,   'data_processed'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('use_proc'     ,   True),
    
    ('nfolds'       ,   15),
    ('logm_bin'     ,   0.01),
    ('mbin_frac'    ,   0.008),
    
])

dtype = [
        ('rockstarId','<i8'), ('logmass','<f4'), ('Ngal','<i8'), ('logsigv','<f4'),
        ('in_train', '?'), ('in_test', '?'), ('fold', '<i4')
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
    
    data = np.ndarray(shape=(len(cat),), dtype=dtype)

    data['rockstarId'] = cat.prop['rockstarId'].values
    data['logmass'] = np.log10(cat.prop['M200c'].values)
    data['Ngal'] = cat.prop['Ngal'].values
    data['logsigv'] = np.log10(cat.prop['sigv'].values)
    data['in_train'] = in_train
    data['in_test'] = in_test
    data['fold'] = fold
    
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
    preds['in_train'] = dat['in_train']
    preds['in_test'] = dat['in_test']
    
    regr_models = np.ndarray(shape=(dat['fold'].max()+1,), 
                             dtype=[('fold', '<i4'), ('coef','<f4'), 
                                    ('intercept','<f4'), ('R^2','<f4')])

    for i in range(dat['fold'].max()+1):
        print('\nfold #'+str(i))
        
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(dat['logmass'][(dat['fold']!=i) & dat['in_train']].reshape(-1, 1), 
                 dat['logsigv'][(dat['fold']!=i) & dat['in_train']])
                 
        regr_models[i]['coef'] = regr.coef_
        regr_models[i]['intercept'] = regr.intercept_
        regr_models[i]['R^2'] = regr.score(dat['logmass'][(dat['fold']!=i) & dat['in_train']].reshape(-1, 1), 
                                           dat['logsigv'][(dat['fold']!=i) & dat['in_train']])
        
        print('coef:', regr_models[i]['coef'])
        print('intercept:', regr_models[i]['intercept'])
        print('R^2:', regr_models[i]['R^2'])
        
        preds['logmass_pred'][(dat['fold']==i) & (dat['in_test'])] = \
            (dat['logsigv'][(dat['fold']==i) & dat['in_test']] - regr.intercept_)/regr.coef_
        
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
i = 0

f = plt.figure(figsize=[4,6])
gs = mpl.gridspec.GridSpec(2,1,height_ratios=[1,1], hspace=0)

ax1 = f.add_subplot(gs[0,0])

matt.binnedplot( data_pure['logmass'][(data_pure['fold']!=i) & data_pure['in_train']], 
                 data_pure['logsigv'][(data_pure['fold']!=i) & data_pure['in_train']],
                 n=50,
                 percentiles = [35,47.5],
                 ax=ax1,
                 label='pure train',
                 names=False,
                 c='m',
                 log=0
                )
matt.binnedplot( data_contam['logmass'][(data_contam['fold']!=i) & data_contam['in_train']], 
                 data_contam['logsigv'][(data_contam['fold']!=i) & data_contam['in_train']],
                 n=50,
                 percentiles = [35,47.5],
                 ax=ax1,
                 label='contam train',
                 names=False,
                 c='r',
                 log=0
                )

one_to_one = np.arange(11)*(data_contam['logmass'].max() - data_contam['logmass'].min())/10. + data_contam['logmass'].min()

fit_y = one_to_one * regr_pure[i]['coef'] + regr_pure[i]['intercept']
ax1.plot(one_to_one,fit_y,'m',label='pure fit')
fit_y = one_to_one * regr_contam[i]['coef'] + regr_contam[i]['intercept']
ax1.plot(one_to_one,fit_y,'r',label='contam fit')


ax1.set_xlim(xmin=data_contam['logmass'].min(), xmax=data_contam['logmass'].max())
ax1.set_xticks([])

ax1.set_ylabel('$\log[\sigma_v]$', fontsize=14)
ax1.legend(fontsize=8,loc=4)


ax2 = f.add_subplot(gs[1,0],sharex=ax1)
                
ax2.plot(one_to_one,one_to_one,'k',linestyle='dashed')

matt.binnedplot( pred_pure['logmass'][(data_pure['fold']==i) & (data_pure['in_test'])],
                 pred_pure['logmass_pred'][(data_pure['fold']==i) & (data_pure['in_test'])],
                 n=50,
                 percentiles = [35],
                 ax=ax2,
                 label='pure_test',
                 names=False,
                 c='m',
                 log=0
                )
matt.binnedplot( pred_contam['logmass'][(data_contam['fold']==i) & (data_contam['in_test'])],
                 pred_contam['logmass_pred'][(data_contam['fold']==i) & (data_contam['in_test'])],
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

ax1.set_title('$M(\sigma)$; fold #' + str(i))

plt.tight_layout()
f.savefig(os.path.join(model_dir, model_name_save+ '_regr.pdf'))


