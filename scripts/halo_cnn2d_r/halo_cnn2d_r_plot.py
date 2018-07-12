
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

    ('sdm_file'     ,   'data_raw/UM_0.5_MLv_0_preds.npy'),
    ('pure_cat'     ,   'data_mocks/Rockstar_UM_z=0.117_pure.p'),
    ('contam_cat'   ,   'data_mocks/Rockstar_UM_z=0.117_contam.p'),
    
    ('theo_HMF'     ,   'data_raw/dn_dm_MDPL2_z=0.117_M200c.txt'),
    
    ('test_all'     ,   True)
])

print('\n~~~~~ FILE SPECIFICATION ~~~~~')
# Finding model number
data_dir = os.path.join(par['wdir'], 'saved_models', par['model_name'])
log = os.listdir(data_dir)

log = [int(i.split('_')[-1]) for i in log if (i[:-len(i.split('_')[-1])-1]== par['model_name'])]

model_num = 0 if len(log)==0 else max(log)

## Loading data
save_model_name = par['model_name'] + '_' + str(model_num)
model_dir = os.path.join(par['wdir'], 'saved_models', par['model_name'], save_model_name)

print('\nModel Directory: ' + model_dir)


print('\n~~~~~ LOADING ML RESULTS ~~~~~')
cnn_dat = np.load(  os.path.join(data_dir, save_model_name, save_model_name + '.npy'), 
                encoding='latin1').item()
cnn_par = cnn_dat['params']

print('\n~~~~~ LOADING PROCESSED DATA ~~~~~')
data_path = os.path.join(par['wdir'], 'data_processed', par['model_name'])
with open(os.path.join(data_path, par['model_name'] + '.p'), 'rb') as f:
    data_proc = pickle.load(f)['data']

print('\n~~~~~ LOADING SDM RESULTS ~~~~~')
sdm_dat = matt.parseout( np.load(os.path.join(par['wdir'], par['sdm_file'])) )

print('\n~~~~~ LOADING CATALOGS ~~~~~')
pure_cat = Catalog().load(os.path.join(par['wdir'], par['pure_cat']))
contam_cat = Catalog().load(os.path.join(par['wdir'], par['contam_cat']))

rot=0
pure_cat = pure_cat[pure_cat.prop.index[pure_cat.prop['rotation'] == 0].values]
contam_cat = contam_cat[contam_cat.prop.index[contam_cat.prop['rotation'] == 0].values]

print('\n~~~~~ LOADING THEORETICAL HMF ~~~~~')
hmf_M200c = np.loadtxt(os.path.join(par['wdir'], par['theo_HMF']))

x_hmf_M200c, y_hmf_M200c = hmf_M200c

y_hmf_M200c = x_hmf_M200c*y_hmf_M200c*np.log(10)
x_hmf_M200c = np.log10(x_hmf_M200c)


print('\n~~~~~ POWER LAW PREDICTIONS ~~~~~')

if par['test_all']:
    pure_train = list(range(len(pure_cat)))
    pure_test = pure_train
    
    contam_train = list(range(len(contam_cat)))
    contam_test = contam_train
else:
    ind = list(range(len(pure_cat)))
    np.random.shuffle(ind)
    pure_test, pure_train = np.split(ind, [int(len(pure_cat)/10.)])

    ind = list(range(len(contam_cat)))
    np.random.shuffle(ind)
    contam_test, contam_train = np.split(ind, [int(len(contam_cat)/10.)])

pure_regr = linear_model.LinearRegression(fit_intercept=True)
pure_regr.fit(np.log10(pure_cat.prop.loc[pure_train, 'M200c']).values.reshape(-1,1), 
              np.log10(pure_cat.prop.loc[pure_train, 'sigv']))

print('PURE CAT')
print('regr coef: ' +  str(pure_regr.coef_))
print('regr intercept: ' + str(pure_regr.intercept_))

# print('regr R^2: ' + str(pure_regr.score(pure_cat.prop.loc[pure_train, 'M200c'], 
#                                          pure_cat.prop.loc[pure_train, 'sigv'])))

contam_regr = linear_model.LinearRegression(fit_intercept=True)
contam_regr.fit(np.log10(contam_cat.prop.loc[contam_train, 'M200c']).values.reshape(-1,1), 
                np.log10(contam_cat.prop.loc[contam_train, 'sigv']))

print('\nCONTAM CAT')
print('regr coef: ' +  str(contam_regr.coef_))
print('regr intercept: ' + str(contam_regr.intercept_))
# print('regr R^2: ' + str(contam_regr.score(contam_cat.prop.loc[contam_train, 'M200c'], 
#                                            contam_cat.prop.loc[contam_train, 'sigv'])))
                                           

pure_pred = (np.log10(pure_cat.prop.loc[pure_test, 'sigv']) - pure_regr.intercept_) / pure_regr.coef_
contam_pred = (np.log10(contam_cat.prop.loc[contam_test, 'sigv']) - contam_regr.intercept_) / contam_regr.coef_


print('\n~~~~~ PLOTTING POWER LAW PREDICTIONS ~~~~~')
f = plt.figure(figsize=[4,6])
gs = mpl.gridspec.GridSpec(2,1,height_ratios=[1,1], hspace=0)

ax1 = f.add_subplot(gs[0,0])


matt.binnedplot( np.log10(pure_cat.prop.loc[pure_train, 'M200c']), 
                 np.log10(pure_cat.prop.loc[pure_train, 'sigv']),
                 n=50,
                 percentiles = [35,47.5],
                 ax=ax1,
                 label='pure train',
                 names=False,
                 c='m',
                 log=0
                )
matt.binnedplot( np.log10(contam_cat.prop.loc[contam_train, 'M200c']), 
                 np.log10(contam_cat.prop.loc[contam_train, 'sigv']),
                 n=50,
                 percentiles = [35,47.5],
                 ax=ax1,
                 label='contam train',
                 names=False,
                 c='r',
                 log=0
                )

one_to_one = np.arange(11)*(cnn_par['logmass_max'] - cnn_par['logmass_min'])/10. + cnn_par['logmass_min']

fit_y = one_to_one * pure_regr.coef_ + pure_regr.intercept_
ax1.plot(one_to_one,fit_y,'m',label='pure fit')
fit_y = one_to_one * contam_regr.coef_ + contam_regr.intercept_
ax1.plot(one_to_one,fit_y,'r',label='contam fit')


ax1.set_xlim(xmin=cnn_par['logmass_min'], xmax=cnn_par['logmass_max'])
ax1.set_xticks([])

ax1.set_ylabel('$\log[\sigma_v]$', fontsize=14)
ax1.legend(fontsize=8,loc=4)

ax2 = f.add_subplot(gs[1,0],sharex=ax1)
                
ax2.plot(one_to_one,one_to_one,'k',linestyle='dashed')

matt.binnedplot( np.log10(pure_cat.prop.loc[pure_test, 'M200c']),
                 pure_pred,
                 n=50,
                 percentiles = [35],
                 ax=ax2,
                 label='pure_test',
                 names=False,
                 c='m',
                 log=0
                )
matt.binnedplot( np.log10(contam_cat.prop.loc[contam_test, 'M200c']),
                 contam_pred,
                 n=50,
                 percentiles = [35],
                 ax=ax2,
                 label='contam_test',
                 names=False,
                 c='r',
                 log=0
                )
                
ax2.set_xlim(xmin=cnn_par['logmass_min'], xmax=cnn_par['logmass_max'])
ax2.set_ylim(ymin=cnn_par['logmass_min'], ymax=cnn_par['logmass_max'])

ax2.set_xlabel('$\log[M$]', fontsize=14)
ax2.set_ylabel('$\log[M_{pred}$]', fontsize=14)
ax2.legend(fontsize=8, loc=4)

ax1.set_title('Power Law Predictions')

plt.tight_layout()
f.savefig(os.path.join(model_dir, save_model_name+ '_regr.pdf'))


print('\n~~~~~ PLOTTING SDM RESULTS ~~~~~')
f,ax = plt.subplots(figsize=[4,4])

ax.plot(one_to_one,one_to_one,'k',linestyle='dashed')
matt.binnedplot(sdm_dat[:,1],
                sdm_dat[:,0],
                n=50,
                percentiles = [35,47.5],
                ax=ax,
                label='sdm_',
                names=True,
                c='g',
                log=0
               )

ax.set_xlim(xmin=cnn_par['logmass_min'], xmax=cnn_par['logmass_max'])

ax.set_xlabel('$\log[M$]', fontsize=14)
ax.set_ylabel('$\log[M_{pred}$]', fontsize=14)
ax.legend(fontsize=8, loc=4)

ax.set_title('SDM Predictions')

plt.tight_layout()
f.savefig(os.path.join(model_dir, save_model_name+ '_sdm.pdf'))



print('\n~~~~~ CALCULATING MASS ERROR ~~~~~')

pred_err = (10.**cnn_dat['logmass_pred'])/(10.**cnn_dat['logmass_test']) - 1.
sdm_err = (10.**sdm_dat[:,0])/(10.**sdm_dat[:,1]) - 1.

pure_regr_err = (10.**pure_pred)/pure_cat.prop.loc[pure_test, 'M200c'] - 1.
contam_regr_err = (10.**contam_pred)/contam_cat.prop.loc[contam_test, 'M200c'] - 1.


print('\n~~~~~ PLOTTING CNN RESULTS ~~~~~')

f = plt.figure(figsize=[4.5,7])
gs = mpl.gridspec.GridSpec(3,1,height_ratios=[2.5,1.5,1], hspace=0)

ax1 = f.add_subplot(gs[0,0])
ax1.plot(one_to_one,one_to_one, color='k', linestyle='dashed')

matt.binnedplot(cnn_dat['logmass_test'], cnn_dat['logmass_pred'], n=50, 
            percentiles=[34,47.5], median=True, ax=ax1, log=0, c='b'
            )
ax1.set_ylabel(r'$\log[M_{pred}$]',fontsize=14)
ax1.legend(fontsize=8,loc=2)

ax1.set_xticklabels([])
ax1.set_xlim(xmin=cnn_par['logmass_min'], xmax=cnn_par['logmass_max'])


ax2 = f.add_subplot(gs[1,0])# , sharex=ax1)
ax2.plot(one_to_one,[0]*len(one_to_one), color='k', linestyle='dashed')

matt.binnedplot(np.log10(pure_cat.prop.loc[pure_test, 'M200c']),
                pure_regr_err,
                n=25, percentiles=[34], median=True, ax=ax2, 
                label='pure $M(\sigma)$',c='m', errorbar=False, names=False, log=0)
                
"""matt.binnedplot(np.log10(contam_cat.prop.loc[contam_test, 'M200c']),
                contam_regr_err,
                n=25, percentiles=[34], median=True, ax=ax2, 
                label='contam $M(\sigma)$',c='r', errorbar=False, names=False, log=0)

matt.binnedplot(sdm_dat[:,1],sdm_err,n=25, percentiles=[34], median=True, ax=ax2, label='sdm',c='g', errorbar=False, names=False, log=0)"""

matt.binnedplot(cnn_dat['logmass_test'],pred_err,n=25, percentiles=[34], median=True, ax=ax2, label='cnn',c='b', errorbar=False, names=False, log=0)

ax2.set_xticklabels([])
ax2.set_xlim(xmin=cnn_par['logmass_min'], xmax=cnn_par['logmass_max'])

ax2.set_ylim(ymin=-1,ymax=2)
ax2.set_ylabel(r'$\epsilon$',fontsize=14)
ax2.legend(fontsize=8)


ax3 = f.add_subplot(gs[2,0]) #, sharex=ax1)

ax3.plot(x_hmf_M200c,y_hmf_M200c, label='theo', c='k')

_ = matt.histplot(data_proc['logmass'][data_proc['in_train']], 
                  n=50, log=1, box=True, label='train', c='g', ax=ax3)
_ = matt.histplot(data_proc['logmass'][data_proc['in_test']], 
                  n=50, log=1, box=True, label='test', c='r', ax=ax3)

ax3.set_ylabel('$dn/d\log(M_{200c})$',fontsize=14)
ax3.legend(fontsize=8, loc=3)

ax3.set_xlim(xmin=cnn_par['logmass_min'], xmax=cnn_par['logmass_max'])


plt.xlabel(r'$\log(M_{200c})$',fontsize=14)
ax1.set_title(save_model_name)

plt.tight_layout()
f.savefig(os.path.join(model_dir, save_model_name + '_pred.pdf'))  


print('All done!')


