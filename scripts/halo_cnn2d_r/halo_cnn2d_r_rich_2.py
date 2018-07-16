
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
    
    ('scale'        ,   7),
    ('Nkde'         ,   100)
])


print('\n~~~~~ LOADING RICHNESS DATA ~~~~~')
data_path = os.path.join(par['wdir'], 'data_processed', par['model_name'])
with open(os.path.join(data_path, par['model_name'] + '_rich.p'), 'rb') as f:
    dict_rich = pickle.load(f)
    
logmass_bounds = dict_rich['logmass_min'], dict_rich['logmass_max']
shape = dict_rich['shape']
meta = dict_rich['meta']
main = dict_rich['main']    

meta_dtype = [('log_mass', '<f4'), ('Ngal','<i8'), ('fold','<i4'), ('true_frac','<f4')]
main_dtype = [('pdfs','<f4', main[0][0].shape ), ('richs', '<f4', main[0][1].shape)]

meta = np.array(meta, dtype=meta_dtype)
main = np.array(main, dtype = main_dtype)

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
f = plt.figure(figsize=[shape[0]*par['scale'],shape[1]*par['scale']])

gs = mpl.gridspec.GridSpec(*shape)

for i in range(len(meta)):
    
    print('Loading keras model...')
    
    model = keras.models.load_model(os.path.join(model_fold_dir, 'fold_' + str(meta[i]['fold']) + '.h5'))

    print('Predicting...')

    logmass_pred = (logmass_bounds[1] - logmass_bounds[0])*model.predict(main[i]['pdfs']) + logmass_bounds[0]

    logmass_err = (10**logmass_pred)/(10**meta['log_mass']) - 1

    print('Plotting...')
    
    ax = f.add_subplot(gs[i%shape[0],int(i/shape[0])])

    matt.binnedplot(main[i]['richs']/meta[i]['Ngal'], logmass_err,
                    n=20, percentiles = [34,47.5],
                    median = True, ax=ax, log=0
                   )
               
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel('Richness Fraction')
    ax.set_ylabel('$\epsilon$')
    ax.set_title('$\log(M_{200c})=$ ' + str(meta[i]['log_mass'])[:5] + ' ; \n $N=$' + str(meta[i]['Ngal']) + ' ; $f_t=$' + str(meta[i]['true_frac'])[:5])

plt.tight_layout()
f.savefig(os.path.join(save_dir, model_name_save, model_name_save + '_rich.pdf'))  




