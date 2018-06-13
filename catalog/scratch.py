# Scratch file to manipulate/reduce .csv data because it's TOO BIG FOR LOGIN NODE MEMORY

import sys
import os
import numpy as np
import pandas as pd


wdir = '/home/mho1/scratch/halo_cnn'

data_file = os.path.join(wdir,
                         'data_raw',
                         'MDPL2_Rockstar_z=0.117_Macc=1e11.csv')

dat_MD = pd.read_csv(data_file)
print('Raw data size: ' + str(sys.getsizeof(dat_MD)/10.**9) + ' GB\n')

print(dat_MD.head())
