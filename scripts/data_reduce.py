import sys
import os
import numpy as np
import numpy.lib.recfunctions as nprf

#filename = 'UM_z=0.117'
#filename = 'UM_z=0.117_sigclip'
#filename = 'MDPL2_large_z=0.117'
filename = 'UM_z=0.117_medcyl'

in_file = '/home/mho1/scratch/halo_cnn/data_raw/' + filename + '.npy'
out_file = '/home/mho1/scratch/halo_cnn/data_reduced/' + filename + '_reduced.npy'

keep = ['Mtot', 'rotation', 'fold', 'Ngal', 'vlos', 'Rproj',
        'name', 'intest', 'intrain', 'redshift','sigmav','R200','Rs',
        'xyproj', 'hostid', 'truememb'
       ]


print('\nLoading data from: ' + in_file)
dat = np.load(in_file)

print('Original file size: ' + str(sys.getsizeof(dat)/10.**9) + ' GB\n')
print('Original data fields: ' + str(dat.dtype.names) + '\n')


print('Keeping: ' + str(keep) + '\n')
print('Reducing dataset...\n')

dat = nprf.drop_fields(dat, [i for i in dat.dtype.names if i not in keep])

print("Kept: " + str(dat.dtype.names))

print('Reduced file size: ' + str(sys.getsizeof(dat)/10.**9) + ' GB\n')

print('Saving file to: ' + out_file)

np.save(out_file, dat)
