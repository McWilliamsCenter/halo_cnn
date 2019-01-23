"""
This is a module to define the Cluster and Catalog classes. These classes make it easier and more efficient to store mock observation data. 
"""

## ~~~~~ IMPORTS ~~~~~~
import pickle
import numpy as np
from numpy import ndarray
import pandas as pd

## ~~~~~ CLASS DEFINITIONS ~~~~~~
class Cluster:
    
    def __init__(self, prop=None, gal=None):
        self.prop = prop
        self.gal = gal

class Catalog:
    def __getitem__(self, key):
        
        if (isinstance(key, int) | isinstance(key, list) | isinstance(key, ndarray)):
            return Catalog(par=self.par,
                           prop = self.prop.iloc[key].reset_index(drop=True),
                           gal = self.gal[key])
        else:
            raise Exception("Unknown key type: " + str(type(key)))
            
    def __len__(self):
        return len(self.prop)
    
    def __init__(self, par=None, prop=None, gal=None):
        self.par = par # Parameters of mock catalog
        self.prop = prop # Properties of host clusters. (# clusters) x (# of properties)
        self.gal = gal # Cluster members. Position, velocities, etc.
    
    def save(self,filename, protocol=4):
        print('Pickle dumping to ' + filename + ' with protocol ' + str(protocol))
        with open(filename, 'wb') as out_file:
            pickle.dump(self, out_file, protocol=protocol)
            
    def save_npy(self, filename):
        print('Saving as npy: ' + filename)
        max_ngal = self.prop['Ngal'].max()

        dtype = [(x,'f') for x in self.prop.columns.values]
        dtype += [('gal_'+x[0],x[1],max_ngal) for x in self.gal[0].dtype.descr]
        
        out = np.zeros(shape=(len(self),), dtype=dtype)
        
        print('Loading prop')
        for x in self.prop.columns.values:
            out[x] = self.prop[x].values
        
        print('Loading gal')
        for i in range(len(self)):
            for f in self.gal[0].dtype.names:
                out[i]['gal_'+f][:int(out[i]['Ngal'])] = self.gal[i][f]
                
        print('Parameters not transferred:')
        print(self.par)
                
        np.save(filename, out)
        
    def load(self, filename):
        print('Loading catalog from: ' + filename)
        with open(filename, 'rb') as in_file:
            new_cat = pickle.load(in_file)
        
        self.par = new_cat.par
        self.prop = new_cat.prop
        self.gal = new_cat.gal
        
        return self
