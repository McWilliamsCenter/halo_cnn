
## IMPORTS

import os
import sys
import numpy as np
from sklearn import linear_model
from scipy.stats import gaussian_kde
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Cluster, Catalog


## DATA PARAMETERS
par = OrderedDict([
    ('model_name'   ,   'halo_cnn1d_r'),

    ('wdir'         ,   '/home/mho1/scratch/halo_cnn'),
    ('in_folder'    ,   'data_mocks')
    ('out_folder'   ,   'data_processed')
    
    ('data_file'    ,   'Rockstar_UM_z=0.117_contam.p'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('shape'        ,   (48,)), # Length of a cluster's ML input array. # of times velocity pdf will be sampled 

])




## DATA
print('\n~~~~~ LOADING DATA ~~~~~')
# Load and organize

in_path = os.path.join(par['wdir'], par['in_folder'], par['data_file'])

cat = Catalog().load(in_path)

# Subsample
print('\n~~~~~ SUBSAMPLING ~~~~~')
    
print('Data length: ' + str(len(cat)))
ind = np.random.choice(range(len(pure_cat)), 
                       int(par['subsample']*len(pure_cat)),
                       replace=False
                      )

cat.prop = cat.prop.iloc[ind]
cat.gal = cat.gal[ind]

print('Subsampled data length: ' + str(len(cat)))



print('\n~~~~~ DATA CHARACTERISTICS ~~~~~')

print(cat.par)





print('\n~~~~~ PREPROCESSING DATA ~~~~~~')

# Generate input data
print('\nGenerate input data')

vpdfs = np.ndarray(shape = (len(dat), *par['shape']))

mesh = np.mgrid[-vmax : vmax : par['shape']*1j]

sample = np.vstack([mesh.ravel()]) # Velocities at fixed intervals. Used to sample velocity pdfs

print('Generating ' + str(len(dat)) + ' KDEs...')
for i in range(len(dat)):
    if i%1000==0: print(str(int(i/1000)) + ' / '+str(int(len(dat)/1000)))

    # initialize a gaussian kde from galaxy velocities
    kde = gaussian_kde(dat['vlos'][i][:dat['Ngal'][i]])
    
    # sample kde at fixed intervals
    kdeval = np.reshape(kde(sample).T, mesh.shape)
    
    # normalize input
    kdeval /= kdeval.sum()
    
    vpdfs[i,:] = kdeval

vpdfs = vpdfs.astype('float32')

print("KDE's generated.")


print('\nConverting masses, sigv to log scale')

masses = np.log10(dat['Mtot'])
masses = masses.reshape((len(masses),1))

sigv = np.log10(dat['sigmav'])
sigv = sigv.reshape((len(masses),1))






## SAVE
print('\nSAVE')

model_dir = os.path.join(par['wdir'], 'data_processed', par['model_name'])

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

print('Writing parameters to file')
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as param_file:
    param_file.write('\n~~~ DATA PARAMETERS ~~~ \n\n')
    for key in par.keys():
        param_file.write(key + ' : ' + str(par[key]) +'\n')
    
    param_file.write('\n\n')

save_dict = {
    'params'    :   par,
    
    'vpdf'      :   vpdfs,
    'mass'      :   masses,
    'sigv'      :   sigv,
    
    'in_train'  :   (dat['intrain']==1),
    'in_test'   :   (dat['intest']==1),
    'fold'      :   dat['fold']
} 


np.save(os.path.join(model_dir, par['model_name'] + '.npy'), save_dict)

print('Data saved.')







## PLOT CHARACTERISTICS
print('\n~~~~~ PLOTTING MASS DIST ~~~~~')


f, ax = plt.subplots(figsize=(5,5))

mass_train = masses[(dat['intrain']==1)]
mass_test = masses[(dat['intest'] == 1)]

_ = matt.histplot(  mass_train, n=75, log=False, 
                    label='train', ax=ax)
_ = matt.histplot(  mass_test, n=75, log=False, 
                    label='test', ax=ax)
    
plt.xlabel('$\log(M_{200c})$', fontsize=20)
plt.ylabel('$N$', fontsize=20)

plt.legend()
plt.tight_layout()
f.savefig(os.path.join(model_dir, par['model_name'] + '_ydist.pdf'))


print('Figures saved')


print('All finished!')
