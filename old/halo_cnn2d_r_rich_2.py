
## IMPORTS

import sys
import os
import numpy as np
import pickle

from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import keras

## FUNCTIONS

import tools.matt_tools as matt


## PARAMETERS
par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_cnn2d_r'),
    
    ('plot_size'    ,   4),
    
    ('scale'        ,    'pred')
])


print('\n~~~~~ LOADING RICHNESS DATA ~~~~~')
data_path = os.path.join(par['wdir'], 'data_processed', par['model_name'])
with open(os.path.join(data_path, par['model_name'] + '_rich.p'), 'rb') as f:
    dict_rich = pickle.load(f)
    
logmass_bounds = dict_rich['logmass_min'], dict_rich['logmass_max']
shape = dict_rich['shape']
meta = dict_rich['meta']
pdfs = dict_rich['pdfs']
richs = dict_rich['richs']

meta_dtype = [('rockstarId','<i8'),('log_mass', '<f4'), ('Ngal','<i8'), ('fold','<i4'), ('true_frac','<f4')]

meta = np.array(meta, dtype=meta_dtype)
# main = np.array(main, dtype = main_dtype)

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



print('\n~~~~~ GENERATING RICHNESS ERROR ~~~~~')
f = plt.figure(figsize=[shape[1]*par['plot_size'],shape[0]*par['plot_size']])

gs = mpl.gridspec.GridSpec(*shape)

margin_err = [[]] * shape[1]
margin_rich = [[]]*shape[1]

for i in range(len(meta)):

    print('rId:', meta[i]['rockstarId'])
    
    # Load keras model
    model = keras.models.load_model(os.path.join(model_fold_dir, 'fold_' + str(meta[i]['fold']) + '.h5'))
    
    # Make predictions
    logmass_pred = (logmass_bounds[1] - logmass_bounds[0])*model.predict(pdfs[i]) + logmass_bounds[0]

    # Calculate mass error
    if par['scale']=='true':
        logmass_err = (10**(logmass_pred - meta[i]['log_mass'])) - 1
    elif par['scale'] =='pred':
        logmass_err = (10**(logmass_pred - logmass_pred[-1])) - 1
    
    ax = f.add_subplot(gs[i%shape[0],int(i/shape[0])])

    matt.binnedplot(richs[i]/meta[i]['Ngal'], logmass_err,
                    n=20, percentiles = [34,47.5],
                    median = True, ax=ax, log=0, c='b'
                   )

    ax.axhline(y=logmass_err[-1], linestyle='dashed', c='k')
    ax.set_xlim(0,1)
    ax.set_ylim(-1, 2*np.abs(logmass_err[-1]+1) - 1)
    ax.set_xlabel('Richness Fraction')
    ax.set_title('$\log(M_{200c})=$ ' + str(meta[i]['log_mass'])[:5] + ' ; \n $N=$' + str(meta[i]['Ngal']) + ' ; $f_t=$' + str(meta[i]['true_frac'])[:5])
    
    if par['scale']=='true':
        ax.set_ylabel('$\epsilon$')
    elif par['scale'] == 'pred':
        ax.set_ylabel('$\epsilon_{1.0}$')
        
   
    margin_err[i%shape[1]] = np.append(margin_err[i%shape[1]], logmass_err)
    margin_rich[i%shape[1]] = np.append(margin_rich[i%shape[1]], richs[i]/meta[i]['Ngal'])
    

plt.tight_layout()
f.savefig(os.path.join(save_dir, model_name_save, model_name_save + '_rich.pdf'))  


print('\n~~~~~ PLOTTING MARGINALIZED ERROR ~~~~~')

f,ax = plt.subplots(figsize=(7,7))

colr = ['b','r','g','c','m','y']

for i in range(shape[1]):

    matt.binnedplot(margin_rich[i], margin_err[i],
                    n=20, percentiles = [34],
                    median = True, ax=ax, log=0,
                    label=str(i), names=False,
                    c = colr[i%len(colr)]
                   )
                   
ax.axhline(y = 0, linestyle='dashed', c = 'k')
ax.set_xlim(0,1)
ax.set_ylim(-1, 1)
ax.set_xlabel('Richness Fraction')

if par['scale']=='true':
    ax.set_ylabel('$\epsilon$')
elif par['scale'] == 'pred':
    ax.set_ylabel('$\epsilon_{1.0}$')

plt.legend()
plt.tight_layout()
f.savefig(os.path.join(save_dir, model_name_save, model_name_save + '_richmargin.pdf'))  




