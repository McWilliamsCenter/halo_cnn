import os
import numpy as np
import pandas as pd
import pickle

from tools.plot_tools import binned_plot
from tools.catalog import Catalog

wdir = '/hildafs/home/mho1/hilda/halo_cnn'
cache_dir = os.path.join(wdir, 'data_mocks', 'MDPL2_UM_sn121_sm9.5_extralarge_031222.cache')

# ~~~ LOAD CONFIG GENERAL FILE ~~~
with open(os.path.join(cache_dir, 'cache.p'), 'rb') as f:
    config = pickle.load(f)
for key in config.keys():
    print(key + ' : ' + str(config[key]))
    
# ~~~ LOAD AND COMBINE INCREMENTAL CATALOGS ~~~
typ = 'contam'

for i in range(0, config['split_rot']):
    try:
        if i==0:
            cat = Catalog().load(
                os.path.join(cache_dir, config['catalog_name']+'_' + typ + f'_{i}.p'))
        else:
            cat = cat + Catalog().load(
                os.path.join(cache_dir, config['catalog_name']+'_' + typ + f'_{i}.p'))
    except FileNotFoundError:
        print(f'File {i} not found')
        pass

# ~~~ RENUMBER ROTATIONS ~~~
rot_angs = np.unique(cat.prop[['rot_theta', 'rot_phi']], axis=0)
rot_num = np.arange(3, len(rot_angs)+3)

rot_num[(rot_angs[:,0] == 0) & (rot_angs[:,1]==0)] = 0
rot_num[(rot_angs[:,0] == np.pi/2) & (rot_angs[:,1]==0)] = 1
rot_num[(rot_angs[:,0] == np.pi/2) & (rot_angs[:,1]==np.pi/2)] = 2

print(rot_num)

index = pd.MultiIndex.from_tuples(list(rot_angs), names=["rot_theta", "rot_phi"])
rot_numS = pd.Series(rot_num, index=index)

cat.prop['rotation'] = rot_numS[list(zip(cat.prop['rot_theta'], cat.prop['rot_phi']))].values


# # ~~~ APPEND M200c etc. INFORMATION ~~~
# print('Loading mass def information ...')
# minfo = pd.read_csv('/hildafs/home/mho1/hilda/halo_cnn/data_raw/uchuu/Uchuu_snap:050_m200c1e11.csv')

# print('Checking subset...')
# left = set(cat.prop['id'].values.astype(int))
# right = set(minfo['id'].values.astype(int))
# print('Mass def has all the catalog halos:', 
#       left.issubset(right)
#       )

# print('Appending mass information...')
# minfo = minfo[['id','M200c','Macc','Mvir']].set_index('id')
# minfo = minfo.loc[cat.prop['id'].values.astype(int)]
# cat.prop = pd.concat([cat.prop, minfo.reset_index(drop=True)], axis=1)

print('Saving...')
cat.save(os.path.join(wdir, 'data_mocks', 'MDPL2_UM_sn121_sm9.5_extralarge_031222_' + typ + '.p'))
