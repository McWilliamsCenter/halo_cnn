

## IMPORTS

import sys
import numpy as np
import numpy.lib.recfunctions as nprf


def load_dat(data_path, keep = ['Mtot', 'rotation', 'fold', 'Ngal', 'vlos', 
                                'Rproj','name', 'intest', 'intrain', 
                                'redshift','sigmav', 'xyproj']):

    print('Loading data...')
    
    dat_orig = np.load(data_path)
    
    print('Data loaded successfully')

    print(str(sys.getsizeof(dat_orig)/10.**9) + ' GB')
    temp = nprf.drop_fields(dat_orig, [i for i in dat_orig.dtype.names if i not in keep])
    print(str(sys.getsizeof(temp)/10.**9) + ' GB')
    del(dat_orig)

    return temp
