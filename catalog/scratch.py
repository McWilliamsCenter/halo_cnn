# Scratch file to manipulate/reduce .csv data because it's TOO BIG FOR LOGIN NODE MEMORY

import sys
import os
import numpy as np
import pandas as pd


wdir = '/home/mho1/scratch/halo_cnn'

MD_file = os.path.join(wdir,
                         'data_raw',
                         'MDPL2_Rockstar_z=0.117_Macc=1e11.csv')
UM_file = os.path.join(wdir,
                         'data_raw',
                         'sfr_catalog_0.895100.npy')
print('Loading MD...')
dat_MD = pd.read_csv(MD_file)
print('MD data size: ' + str(sys.getsizeof(dat_MD)/10.**9) + ' GB\n')

print('Loading UM...')
dat_UM = pd.DataFrame(np.load(UM_file))
print('UM data size: ' + str(sys.getsizeof(dat_UM)/10.**9) + ' GB\n')

print(len(dat_MD))
print(len(dat_UM))


print(dat_MD.head())
print(list(dat_MD.columns))
print(dat_UM.head())
print(list(dat_UM.columns))

hosts_MD = dat_MD[dat_MD['upId']==-1]
hosts_UM = dat_UM[dat_UM['upid']==-1]

print(len(hosts_MD))
print(len(hosts_UM))

for i, host in hosts_UM.head(5).iterrows():
    print(host['id'])
    
    print(hosts_MD[hosts_MD['rockstarId'] == host['id']])

i_MD = list(hosts_MD['rockstarId'].sort_values())
i_UM = list(hosts_UM['id'].sort_values())
"""
nomatch = 0
j=0
for i in range(len(i_UM)):
    if i%10000==0: print(i)
    
    while j<len(i_MD):"""
        
    

print('No match: ',nomatch)
