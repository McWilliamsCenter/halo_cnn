# This file contains the manager for loading and preprocessing halo_cnn data

import os
import sys
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
import pickle
import scipy

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

## FUNCTIONS

import tools.matt_tools as matt
from tools.catalog import Catalog

class HaloCNNDataManager():
	def __init__(self, input_shape,
					   bandwidth=0.25,
					   sample_rate = 1.0,
					   nfolds = 10,
					   mass_range_train = (None, None),
					   mass_range_test = (None, None),
					   dn_dlogm = (10.**-5.2),
					   dlogm = 0.02):
		# initialization
		self.input_shape = input_shape
		self.bandwidth = bandwidth
		self.sample_rate = sample_rate
		self.nfolds = nfolds
		self.mass_range_train = mass_range_train
		self.mass_range_test = mass_range_test
		self.dn_dlogm = dn_dlogm
		self.dlogm = dlogm

		self.loaded = False
		self.vcut = None
		self.aperture = None

		self.dtype = [('id','<i8'), ('logmass', '<f4'), ('Ngal','<i8'), 
					  ('logsigv','<f4'), ('in_train', '?'), ('in_test', '?'), 
					  ('fold', '<i4'), ('pdf', '<f4', self.input_shape)]
		self.data = None

	def make_pdf(self, vlos=[], Rproj=[]):
        return
        
    def _subsample(self, catalog, rate):
        ind = np.random.choice(range(len(cat)), 
                               int(rate*len(cat)),
                               replace=False)
        return cat[ind]
        
    def _assign_traintest(self):
        return
        
    def _assign_folds(self):
        return
        
    def _generate_pdfs(self):
        return
        
	def _preprocess(self, cat):
	    
	    # initialize
	    
	    # subsample
	    
	    # assign test,train
	    
	    # clean
	    
	    # assign folds
	    
	    # generate pdfs
	    
	    return
	    
	def load_catalog(self, catalog, in_train=None, in_test=None, fold=None):
	    
	    return
	    
    def load_catalog_from_file(self, filename):
        catalog = Catalog().load(filename)
        return load(catalog)


    def save(self, filename):
        return
        
    def load(self, filename):
        return


