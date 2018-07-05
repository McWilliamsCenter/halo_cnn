
## IMPORTS

import sys
import os
import numpy as np
import numpy.lib.recfunctions as nprf
# import statsmodels.api as sm
from sklearn import linear_model
from scipy.stats import gaussian_kde

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



## FUNCTIONS

import tools.matt_tools as matt

## MODEL PARAMETERS
wdir = '/home/mho1/scratch/halo_cnn/'

model_name = 'halo_cnn2d_r'
data_dir = os.path.join(wdir, 'saved_models', model_name, 'model_data')

sdm_file = os.path.join(wdir,'data_raw', 'UM_0.5_MLv_0_preds.npy')


# Finding model number
log = os.listdir(data_dir)

log = [int(i.split('_')[-1][:-4]) for i in log if (i[:-len(i.split('_')[-1])-1]== model_name)]

model_num = 0 if len(log)==0 else max(log)

## File Specification
save_model_name = model_name + '_' + str(model_num)
model_dir = os.path.join(wdir, 'saved_models', model_name, save_model_name)

print('\nModel Directory: ' + model_dir)

print('Loading ML results...')
dat = np.load(  os.path.join(data_dir, save_model_name + '.npy'), 
                encoding='latin1').item()
par = dat['params']

print('Loading SDM results...')
sdm_dat = matt.parseout(np.load(sdm_file))





print('Plotting...')

one_to_one = np.arange(11)*(par['mass_max'] - par['mass_min'])/10. + par['mass_min']


print('Power law predictions\n')

y_regr = np.reshape(dat['mass_train'], (len(dat['mass_train']),1))
y_regr_test = np.reshape(dat['mass_test'], (len(dat['mass_test']),1))


regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(y_regr, dat['sigv_regr'])

print('regr coef: ' +  str(regr.coef_))
print('regr intercept: ' + str(regr.intercept_))
print('regr R^2: ' + str(regr.score(y_regr,dat['sigv_regr'])))

y_regr_pred = (dat['sigv_test'] - regr.intercept_)/regr.coef_
y_regr_pred = y_regr_pred.flatten()

f = plt.figure(figsize=[4,6])
gs = mpl.gridspec.GridSpec(2,1,height_ratios=[1,1], hspace=0)

ax1 = f.add_subplot(gs[0,0])

fit_y = one_to_one * regr.coef_ + regr.intercept_
ax1.plot(one_to_one,fit_y,'r',label='fit')

matt.binnedplot(dat['mass_train'],
                 dat['sigv_regr'],
                 n=50,
                 percentiles = [35,47.5],
                 ax=ax1,
                 label='train_',
                 names=True,
                 c='g',
                 log=0
                )
ax1.set_xlim(xmin=par['mass_min'], xmax=par['mass_max'])

ax1.set_ylabel('$\log[\sigma_v]$', fontsize=14)
ax1.legend(fontsize=8,loc=4)

ax2 = f.add_subplot(gs[1,0],sharex=ax1)
                
ax2.plot(one_to_one,one_to_one,'k',linestyle='dashed')
matt.binnedplot(dat['mass_test'],
                 y_regr_pred,
                 n=75,
                 percentiles = [35],
                 ax=ax2,
                 label='test_',
                 names=True,
                 log=0
                )


ax2.set_xlabel('$\log[M$]', fontsize=14)
ax2.set_ylabel('$\log[M_{pred}$]', fontsize=14)
ax2.legend(fontsize=8, loc=4)

ax1.set_title('Power Law Predictions')

plt.tight_layout()
f.savefig(os.path.join(model_dir, save_model_name+ '_regr.pdf'))

print('SDM predictions\n')

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

ax.set_xlim(xmin=par['mass_min'], xmax=par['mass_max'])

ax.set_xlabel('$\log[M$]', fontsize=14)
ax.set_ylabel('$\log[M_{pred}$]', fontsize=14)
ax.legend(fontsize=8, loc=4)

ax.set_title('SDM Predictions')

plt.tight_layout()
f.savefig(os.path.join(model_dir, save_model_name+ '_sdm.pdf'))

print('Calculating mass error\n')

pred_err = (10.**dat['mass_pred'])/(10.**dat['mass_test']) - 1.
regr_err = (10.**y_regr_pred)/(10.**dat['mass_test']) - 1.
sdm_err = (10.**sdm_dat[:,0])/(10.**sdm_dat[:,1]) - 1.


print('Plotting\n')



f = plt.figure(figsize=[4.5,8])
gs = mpl.gridspec.GridSpec(3,1,height_ratios=[2.5,1.5,1], hspace=0)

ax1 = f.add_subplot(gs[0,0])
ax1.plot(one_to_one,one_to_one, color='k', linestyle='dashed')

matt.binnedplot( dat['mass_test'], dat['mass_pred'], n=50, 
            percentiles=[34,47.5], median=True, ax=ax1, log=0
            )
ax1.set_ylabel(r'$\log[M_{pred}$]',fontsize=14)
ax1.legend(fontsize=8,loc=2)

ax1.set_xticklabels([])
ax1.set_xlim(xmin=par['mass_min'], xmax=par['mass_max'])

ax2 = f.add_subplot(gs[1,0])# , sharex=ax1)
ax2.plot(one_to_one,[0]*len(one_to_one), color='k', linestyle='dashed')

# matt.binnedplot(dat['mass_test'],regr_err,n=25, percentiles=[34], median=True, ax=ax2, label='pow',c='r', errorbar=False, names=False, log=0)

matt.binnedplot(dat['mass_test'],pred_err,n=25, percentiles=[34], median=True, ax=ax2, label='cnn',c='b', errorbar=False, names=False, log=0)

matt.binnedplot(sdm_dat[:,1],sdm_err,n=25, percentiles=[34], median=True, ax=ax2, label='sdm',c='g', errorbar=False, names=False, log=0)

ax2.set_xticklabels([])
ax2.set_xlim(xmin=par['mass_min'], xmax=par['mass_max'])

ax2.set_ylim(ymin=-1,ymax=2)
ax2.set_ylabel(r'$\epsilon$',fontsize=14)
ax2.legend(fontsize=8)




ax3 = f.add_subplot(gs[2,0]) #, sharex=ax1)
_ = matt.histplot(dat['mass_train'], n=100, log=1, box=True, label='train', ax=ax3)
_ = matt.histplot(dat['mass_test'], n=100, log=1, box=True, label='test', ax=ax3)


ax3.set_ylabel('$dn/d\log(M_{200c})$',fontsize=14)
ax3.legend(fontsize=8, loc=3)

ax1.set_xlim(xmin=par['mass_min'], xmax=par['mass_max'])

plt.xlabel(r'$\log(M_{200c})$',fontsize=14)
ax1.set_title(model_name)

plt.tight_layout()
f.savefig(os.path.join(model_dir, save_model_name + '_pred.pdf'))

print('All done!')
