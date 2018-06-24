import sys
import os
import pandas as pd

wdir = '/home/mho1/scratch/halo_cnn'

orig_file = os.path.join(wdir,'data_query','MDPL2_Rockstar_snap:120_v2.csv')
reduced_file = os.path.join(wdir,'data_query','MDPL2_Rockstar_snap:120_v2_reduced.csv')

print('\nLoading original data...')
dat = pd.read_csv(orig_file,  index_col=0)

print('Original data size: ' + str(sys.getsizeof(dat)/10.**9) + ' GB\n')

new_dat = dat[
    (dat['x'] < 500) &
    (dat['y'] < 500) &
    (dat['z'] < 500)
]

del dat

print('New data size: ' + str(sys.getsizeof(new_dat)/10.**9) + ' GB\n')

new_dat.to_csv(reduced_file)
