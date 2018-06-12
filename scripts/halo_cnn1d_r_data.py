
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


## DATA PARAMETERS
par = OrderedDict([

    ('wdir'         ,   '/home/mho1/scratch/halo_cnn/'),
    ('model_name'   ,   'halo_cnn1d_r'),
    ('data_file'    ,   'UM_z=0.117_med_reduced.npy'),
    
    ('subsample'    ,   1.0 ), # Fraction by which to randomly subsample data
    
    ('shape'        ,   48), # Length of a cluster's ML input array. # of times velocity pdf will be sampled 
    
    ('new_train'    ,   False) # Reassign training data to have a flat mass distribution. For use for training with one fold

])




## DATA
print('\n~~~~~ LOADING DATA ~~~~~')
# Load and organize

raw_path = os.path.join(par['wdir'], 'data_reduced')

print('\nData file: ' + par['data_file'] + '\n')

dat_orig = np.load(os.path.join(raw_path, par['data_file']))
print('Raw data size: ' + str(sys.getsizeof(dat_orig)/10.**9) + ' GB\n')

# Subsample
print('\n~~~~~ SUBSAMPLING ~~~~~')
    
print('Original data length: ' + str(len(dat_orig)))
dat = np.random.choice( dat_orig, 
                        int(par['subsample']* len(dat_orig)),
                        replace = False)    
print('New data length: ' + str(len(dat)))


# Reassign train datapoints to an even cluster mass distribution
if par['new_train']:
    print('\nREASSIGN TRAINING DATA')

    intrvl = 0.01
    num_train = 4000

    dat['intrain']=0

    logM = np.log10(dat['Mtot'])

    kde = gaussian_kde(logM) # get a probability distribution for logM

    sample = np.arange(logM.min(),
                       logM.max(),
                       intrvl
                      )
    pdf = kde(sample)
    pdf /= pdf.sum()
    
    pdf = 1./pdf # invert it
    pdf /= pdf.sum()

    # sample pdf at each datapoint
    p = np.ndarray(shape=len(logM))

    for i in range(len(logM)):
        p[i] = pdf[int((logM[i] - logM.min())/intrvl)] 

    p/=p.sum()


    new_logM_indices = np.random.choice(np.arange(len(logM)), num_train, replace=False, p=p) # use the probabilities to choose an even mass profile
    
    
    dat['intrain'][new_logM_indices]=1




print('\n~~~~~ DATA CHARACTERISTICS ~~~~~')
print('data shape: ' + str(dat.shape))

vmax = dat['vlos'].max()
rmax = dat['Rproj'].max()

par['vmax'] = vmax
par['rmax'] = rmax

print('vmax: ' + str(vmax))
print('rmax: ' + str(rmax))






print('\n~~~~~ PREPROCESSING DATA ~~~~~~')

# Generate input data
print('\nGenerate input data')

vpdfs = np.ndarray(shape = (len(dat), par['shape']))

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
