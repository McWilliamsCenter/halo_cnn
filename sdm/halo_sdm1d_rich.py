
## IMPORTS

import sys
import os
import numpy as np
import time
import pickle
import multiprocessing as mp
import pandas as pd
import sdm
from sklearn.cross_validation import KFold

from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Catalog


## ML PARAMETERS
par = OrderedDict([ 
	('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
	('model_name'   ,   'halo_sdm1d_rich'),

	('in_folder'    ,   'data_mocks'),
	('data_file'    ,   'Rockstar_UM_z=0.117_contam_med.p'),

	('subsample'    ,   0.1),
	('nfolds'       ,   10),
	
	('test_range'   ,   (10**13.9, 10**15.1)),
	
	('dn_dlogm'		,	10.**-5.2),
	('dlogm'		,	0.02),


	('sdrModel'		,	'NuSDR'),
	('K'			,	4),
	('divfunc'		,	'kl'),
	('progressbar'  ,   False),

	('n'			,	50),
	('per_clus'		,	10)

])


n_proc_model=60
n_proc_pool=1
np.random.seed(111523)
debug=True


print('\n~~~~~ MODEL PARAMETERS ~~~~~')
for key in par.keys():
	print(key, ':', str(par[key]))



## DATA
print('\n~~~~~ LOADING DATA ~~~~~')
# Load and organize

in_path = os.path.join(par['wdir'], par['in_folder'], par['data_file'])

cat = Catalog().load(in_path)

if (cat.par['vcut'] is None) & (cat.par['aperture'] is None):
	print('Pure catalog...')
	cat.par['vcut'] = 3785.
	cat.par['aperture'] = 2.3





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

# Subsample
print('\n~~~~~ SUBSAMPLING ~~~~~')
	
print('Data length: ' + str(len(cat)))

if par['subsample'] < 1:
	ind = np.random.choice(range(len(cat)), 
						   int(par['subsample']*len(cat)),
						   replace=False
						  )
	cat = cat[ind]
	in_train = in_train[ind]
	in_test = in_test[ind]
	fold = fold[ind]

print('Subsampled data length: ' + str(len(cat)))



## RUN SDM

import datetime
start_t = time.time()

print("\n~~~ start: %s ~~~" % str(datetime.datetime.now()))

# Declare model
if par['sdrModel'] == 'NuSDR':
	model = sdm.NuSDR(n_proc = n_proc_model, div_func = par['divfunc'], K=par['K'], progressbar=par['progressbar'])
elif par['sdrModel'] == 'SDR':
	model = sdm.SDR(n_proc = n_proc_model, div_func = par['divfunc'], K=par['K'], progressbar=par['progressbar'])
else:
	raise Exception('unkown sdm model')


i=0

# get scaler
train_ind = np.argwhere(in_train & (fold!=i)).flatten()
x_train = [[(j,) for j in np.abs(cat.gal[m]['vlos'])] for m in train_ind]
y_train = np.log10(cat.prop['M200c'][in_train & (fold!=i)])
feats_train = sdm.Features(x_train, mass=y_train, default_category='train')
feats_train, scaler = feats_train.standardize(ret_scaler=True)

def train(x):

	# Train/test indices
	train_ind = np.argwhere(in_train & (fold!=i)).flatten()
	test_ind = np.argwhere(in_test & (fold==i)).flatten()

	# 3-fold cross-vold parameter tuning
	halo_folds = KFold(n=len(set(train_ind)), n_folds=3, shuffle=True)
	model._tune_folds = [[np.vectorize(x.__contains__)(fold[in_train & (fold!=i)]) for x in traintest] \
							 for traintest in halo_folds]

	# Generate train/test features
	x_train = [[(j,1) for j in np.abs(cat.gal[m]['vlos'])] for m in train_ind]
	y_train = np.log10(cat.prop['M200c'][in_train & (fold!=i)])

	x_test = [[(j,1) for j in np.abs(cat.gal[m]['vlos'])] for m in test_ind]
	y_test = np.log10(cat.prop['M200c'][in_test & (fold==i)])

	x_test = x_test*par['per_clus']
	y_test = list(y_test)*par['per_clus']

	sample_test = [int((x+ np.random.random()*1./(par['n']-1))*len(m))/len(m) for \
					m in x_test] if x<1 else np.ones(len(x_test))
	x_test = [list(np.array(x_test[m])[np.random.choice(len(x_test[m]), 
														int(sample_test[m]*len(x_test[m])),
														replace=False)]) for m in range(len(x_test))]
	test_ind_corr = [m for m in range(len(x_test)) if len(x_test[m])>par['K']]
	x_test = np.array(x_test)[test_ind_corr]
	y_test = np.array(y_test)[test_ind_corr]
	sample_test = np.array(sample_test)[test_ind_corr]

	if len(x_test)==0:
		return np.zeros((0,4),'f')

	feats_train = sdm.Features(x_train, mass=y_train, default_category='train')
	feats_train = feats_train.standardize(scaler=scaler)

	feats_test = sdm.Features(x_test, default_category='test')
	feats_test = feats_test.standardize(scaler=scaler)

	# Train and predict
	preds = model.transduct(feats_train, feats_train.mass, feats_test, save_fit=True)

	out = np.zeros((len(preds),4),'f')

	out[:,0] = y_test
	out[:,1] = preds
	out[:,2] = i
	out[:,3] = sample_test

	return out

import tqdm
if n_proc_pool > 1:
	with mp.Pool(processes=n_proc_pool) as pool:
		outs = np.array(list( tqdm.tqdm(pool.imap(train, np.linspace(1,0,par['n'])), total=par['n']) ))
else:
	outs = list( tqdm.tqdm(map(train, np.linspace(1,0,par['n'])), total=par['n']) )

print('~~~~~ DONE ~~~~~')
print("Training time: %s hours" % ((time.time() - start_t) / (60*60) ))

print('~~~~~ COMBINE OUTPUTS ~~~~~')
out_full = np.zeros((0,4),'f')

for j in outs:
	out_full = np.append(out_full, j, axis=0)


print('~~~~~ FILE SPECIFICATION ~~~~~')
## FILE SPECIFICATION

save_dir = os.path.join(par['wdir'], 'saved_models', par['model_name'])

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

# Assign most recent model number
log = os.listdir(save_dir)

log = [int(i.split('_')[-1]) for i in log if (i[:-len(i.split('_')[-1])-1]== par['model_name'])]

par['model_num'] = 0 if len(log)==0 else max(log) + 1

model_name_save = par['model_name'] + '_' + str(par['model_num'])
model_dir = os.path.join(save_dir, model_name_save)
if not os.path.isdir(model_dir):
	os.makedirs(model_dir)

print(model_dir)

print('~~~~~ SAVING OUTPUT DATA ~~~~~')

save_dict = {
	'params'    :   par,

	'logmass_test'    :   out_full[:,0],
	'logmass_pred'    :   out_full[:,1],
	'fold'            :   out_full[:,2],
	'x'				  :	  out_full[:,3]
}

np.save(os.path.join(model_dir, model_name_save + '.npy'), save_dict)

print('Output data saved.')
	

