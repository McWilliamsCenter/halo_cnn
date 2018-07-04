
""" 
TODO: PARALLELIZE APERATURE, VCUT
"""


import sys
import os

import pickle
import numpy as np
import pandas as pd

from collections import OrderedDict, defaultdict

from tools.catalog import Catalog


## ~~~~~~ PARAMETERS ~~~~~~
par = OrderedDict([ 
    ('wdir'         ,   '/home/mho1/scratch/halo_cnn'),
    ('in_folder'    ,   'data_mocks'),
    ('out_folder'   ,   'data_mocks'),
    
    ('in_file'      ,   'Rockstar_UM_z=0.117_contam.p'),
    ('out_file'     ,   'Rockstar_UM_z=0.117_contam_rot10.p'),
    
    ('min_mass'     ,   10**(13.5)),
    ('min_richness' ,   10),
    
    ('cut_size'     ,   'large'),
    ('rotations'    ,   10),

    ])
    

## ~~~~~ PARAMETER HANDLING ~~~~~

if par['cut_size'] == 'small':
    aperture = 1.1     #the radial aperture in comoving Mpc/h
    vcut = 1570.       #maximum velocity is vmean +/- vcut in km/s
    
elif par['cut_size'] == 'medium':
    aperture = 1.6 #Mpc/h
    vcut = 2500. #km/s
    
elif par['cut_size'] == 'large':
    aperture = 2.3 #Mpc/h
    vcut = 3785. #km/s
else:
    raise Exception('Invalid cut_size')
    

## ~~~~~ LOADING DATA ~~~~~
print('\n~~~~~ LOADING DATA ~~~~~')

in_path = os.path.join(par['wdir'], par['in_folder'], par['in_file'])

cat = Catalog().load(in_path)

print('\n~~~~~ ORIGINAL PARAMS ~~~~~')
for key in cat.par.keys():
    print(key + ' : ' + str(cat.par[key]))
    
## ~~~~~ MAKING CHANGES ~~~~~
print('\n~~~~~ MAKING CHANGES ~~~~~')

# min_mass
if par['min_mass'] > cat.par['min_mass']:
    cat = cat[ cat.prop.index[cat.prop['M200c'] >= par['min_mass']].values ]
    cat.par['min_mass'] = par['min_mass']
    
    print('new min_mass: ' + str(cat.par['min_mass']))
else:
    print('min_mass unchanged')


# min_richness
if par['min_richness'] > cat.par['min_richness']:
    cat = cat[ cat.prop.index[cat.prop['Ngal'] >= par['min_richness']].values ]
    cat.par['min_richness'] = par['min_richness']
    
    print('new min_richness: ' + str(cat.par['min_richness']))
else:
    print('min_richness unchanged')
    

# rotations
if par['rotations'] < (cat.prop['rotation'].max() + 1):
    cat = cat[ cat.prop.index[cat.prop['rotation'] < par['rotations']].values ]
    
    print('new #rotations: ' + str(par['rotations']))
else:
    print('#rotations unchanged')
    

# note if pure
if (cat.par['aperture'] is None) | (cat.par['vcut'] is None):
    print('\n>>>>> PURE CATALOG <<<<<')

recalc_stat = False

# aperture
if (cat.par['aperture'] is None) | (aperture < cat.par['aperture']):
    print('Changing aperature...')
    for i in range(len(cat)):
        if i%int(len(cat)/100) == 0: print(i, '/100')
        
        cat.gal[i] = cat.gal[i][cat.gal[i]['Rproj'] < aperture]
        
        cat.prop.loc[i,'Ngal'] = len(cat.gal[i])
    
    cat = cat[ cat.prop.index[cat.prop['Ngal'] > cat.par['min_richness']].values ]
    cat.par['aperture'] = aperture
    
    recalc_stat = True
    
    print('new aperture: ' + str(cat.par['aperature']))
else:
    print('aperture unchanged')
    
# vcut
if (cat.par['vcut'] is None) | (vcut < cat.par['vcut']):
    print('Changing vcut...')
    for i in range(len(cat)):
        if i%int(len(cat)/100) == 0: print(i, '/100')
        
        cat.gal[i] = cat.gal[i][np.abs(cat.gal[i]['vlos']) < vcut]
        
        cat.prop.loc[i,'Ngal'] = len(cat.gal[i])
    
    cat = cat[ cat.prop.index[cat.prop['Ngal'] > cat.par['min_richness']].values ]
    cat.par['vcut'] = vcut
    
    recalc_stat = True
    
    print('new vcut: ' + str(cat.par['vcut']))
else:
    print('vcut unchanged')
    
if recalc_stat:    
    print('\n~~~~~ RECALCULATING STATISTICS ~~~~~')

    cat.prop['sigv'] = [np.std(x['vlos']) for x in cat.gal]

print('New data len:', len(cat))

print('\n~~~~~ SAVING ~~~~~')

cat.save(os.path.join(par['wdir'], 
                      par['out_folder'], 
                      par['out_file']))

print('All done.')


