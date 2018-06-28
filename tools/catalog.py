"""
This is a module to define the Cluster and Catalog classes. These classes make it easier and more efficient to store mock observation data. 
"""

## ~~~~~ IMPORTS ~~~~~~
import pickle

## ~~~~~ CLASS DEFINITIONS ~~~~~~
class Cluster:
    
    def __init__(self, prop=None, gal=None):
        self.prop = prop
        self.gal = gal

class Catalog:
    def __getitem__(self, key):
        return Cluster(prop = self.prop.iloc[key],
                       gal = self.gal[key])
    def __len__(self):
        return len(self.prop)
    
    def __init__(self, par=None, prop=None, gal=None):
        self.par = par # Parameters of mock catalog
        self.prop = prop # Properties of host clusters. (# clusters) x (# of properties)
        self.gal = gal # Cluster members. Position, velocities, etc.
    
    def save(self,filename):
        with open(filename, 'wb') as out_file:
            pickle.dump(self, out_file)
    def load(self, filename):
        print('Loading catalog from: ' + filename)
        with open(filename, 'rb') as in_file:
            new_cat = pickle.load(in_file)
        
        self.par = new_cat.par
        self.prop = new_cat.prop
        self.gal = new_cat.gal
        
        return self
