# This file contains the manager for loading and preprocessing
# halo_cnn data

import os
import multiprocessing as mp
import pickle
from functools import partial, wraps

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import gaussian_kde


# FUNCTIONS
from tools.catalog import Catalog


# decorators
def check_reg_limits(func):
    wraps(func)

    def wrapper(self, *args, **kwargs):
        if self.reg_limits is None:
            raise Exception('No regularization limits set.')
        return func(self, *args, **kwargs)
    return wrapper

# base classes


class BaseManager():
    def __init__(self, input_shape=None, bandwidth=0.25, 
                 vcut=None, aperture=None,
                 sample_rate=1.0, nfolds=10, val_split=0.,
                 mass_range_train=(None, None), mass_range_test=(None, None),
                 type_train='flat', type_test='all', rot_max=3,
                 dn_dlogm=10.**-5.2, dlogm=0.02, volume=1000**3,
                 *args, **kwargs):
        # KDE parameters
        self.input_shape = input_shape
        self.bandwidth = bandwidth
        self.vcut = vcut
        self.aperture = aperture

        # catalog parameters
        self.sample_rate = sample_rate
        self.nfolds = nfolds
        self.val_split = val_split

        # train/test parameters
        self.mass_range_train = mass_range_train
        self.mass_range_test = mass_range_test
        self.type_train = type_train
        self.type_test = type_test
        self.rot_max = rot_max
        self.dn_dlogm = dn_dlogm
        self.dlogm = dlogm
        self.volume = volume

        # extra
        self.__init_spec__(*args, **kwargs)

    def __init_spec__(self):
        self.bin_limits=None
        self.nbins=None
        return

    def load_catalog(self, filename, cache=True):
        """Loads a catalog file from disk and saves vcut,aperture settings"""
        catalog = Catalog().load(filename)

        if cache:
            self.vcut = catalog.par['vcut']
            self.aperture = catalog.par['aperture']

        return catalog

    def _update_mass_ranges(self, catalog):
        """
            Update mass ranges for test and train sets. If values are None,
            set to max/min of catalog.
        """
        self.mass_range_train = list(self.mass_range_train)
        self.mass_range_test = list(self.mass_range_test)

        if self.mass_range_train[0] is None:
            self.mass_range_train[0] = catalog.prop['M200c'].min()
        if self.mass_range_train[1] is None:
            self.mass_range_train[1] = catalog.prop['M200c'].max()

        if self.mass_range_test[0] is None:
            self.mass_range_test[0] = catalog.prop['M200c'].min()
        if self.mass_range_test[1] is None:
            self.mass_range_test[1] = catalog.prop['M200c'].max()

        self.mass_range_train = tuple(self.mass_range_train)
        self.mass_range_test = tuple(self.mass_range_test)

    def _subsample(self, catalog, rate):
        """Randomly subsample cluster catalog"""
        ind = np.random.choice(range(len(catalog)),
                               int(rate*len(catalog)),
                               replace=False)
        return catalog[ind]

    def _assign_all(self, catalog, mass_range, rot_max):
        """Select all mocks from catalog under a certian rotation maximum"""
        in_array = np.array([False]*len(catalog))

        in_array[(catalog.prop['rotation'] < rot_max) &
                 (catalog.prop['M200c'] > mass_range[0]) &
                 (catalog.prop['M200c'] < mass_range[1])] = True

        return in_array

    def _assign_flat(self, catalog, mass_range):
        """Select a flat mass distribution from catalog"""
        in_array = np.array([False]*len(catalog))

        bin_edges = np.arange(np.log10(mass_range[0]) * 0.9999,
                              (np.log10(mass_range[1]) + self.dlogm)*1.0001,
                              self.dlogm)
        n_per_bin = int(self.dn_dlogm*self.volume*self.dlogm)

        for j in range(len(bin_edges)):
            in_bin = (np.log10(catalog.prop['M200c']) >= bin_edges[j]) & \
                     (np.log10(catalog.prop['M200c']) < bin_edges[j]
                      + self.dlogm)

            if np.sum(in_bin) <= n_per_bin:
                in_array[in_bin] = True  # Assign train members
            else:
                in_array[np.random.choice(np.where(in_bin)[0],
                                          n_per_bin, replace=False)] = True

        return in_array

    def _assign_traintest(self, catalog, set_type, mass_range):
        """Assign cluster mocks to training and test sets."""
        if set_type == 'flat':
            in_array = self._assign_flat(catalog, mass_range)
        elif set_type == 'all':
            in_array = self._assign_all(catalog, mass_range, self.rot_max)
        else:
            raise Exception('Unknown traintest type:' + str(set_type))

        return in_array

    def _assign_val(self, catalog, in_test):
        """Assign a subset of test cluster mocks to validation set."""
        ids_test = catalog.prop['rockstarId'][in_test].drop_duplicates()
        ids_test = np.random.choice(ids_test,
                                    int((1.-self.val_split)*len(ids_test)),
                                    replace=False)
        in_test_new = in_test & \
            catalog.prop['rockstarId'].isin(ids_test).values
        in_val = in_test & ~in_test_new

        return in_test_new, in_val

    def _assign_folds(self, catalog):
        """Use rank-ordering to assign folds evenly for all masses"""
        if self.nfolds == 1:
            return np.full(len(catalog), 1)
        ids_sorted = catalog.prop[['rockstarId', 'M200c']].drop_duplicates()
        ids_sorted = ids_sorted.sort_values(['M200c'])['rockstarId']
        fold_ind = pd.Series(np.arange(len(ids_sorted)) % self.nfolds,
                             index=ids_sorted)
        folds = fold_ind[catalog.prop['rockstarId']].values

        return folds

    def sample_kde(self, data, sample, reweightR=None):
        """
            Generates a Kernel Density Estimator (KDE) using input data and
            samples them at specified points. Handles input and output array
            shape manipulation to ensure 'sample' and output are same shape.

            Args:
                data: input data array to initialize KDE of shape (# of data,
                # of dims)
                sample: sampling array to sample KDE of arbitrary shape
                (*{sample array shape}, # of dims)

            Returns:
                sample_kdeval: output array of same shape as sample array
                (*{sample array shape}, 1)
        """
        ## FIX
        if reweightR is not None:
            kde_reweight = gaussian_kde(reweightR, self.bandwidth)
            kdeR = gaussian_kde(data[:,1], self.bandwidth)
            
            weight = kde_reweight(data[:,1])/kdeR(data[:,1])

            if sample.shape[-1]==1:
                data = data[:,0]

            kde = gaussian_kde(data.T, self.bandwidth, weights=weight)
            
#         elif data.shape[-1] != sample.shape[-1]:
#             raise Exception('Data of dimension %(data_dim)s does not align'
#                             ' with sample dimension %(samp_dim)s' %
#                             {'data_dim': data.shape[-1],
#                              'samp_dim': sample.shape[-1]})

        else:
            if sample.shape[-1]==1:
                data = data[:,0]
            kde = gaussian_kde(data.T, self.bandwidth)

        sample_flat = np.transpose(sample, (len(sample.shape)-1,
                                            *np.arange(len(sample.shape)-1)))
        sample_flat = np.reshape(sample_flat,
                                 (sample.shape[-1],
                                  np.product(sample.shape[:-1])))

        sample_kdeval = kde.evaluate(sample_flat)

        sample_kdeval = np.reshape(sample_kdeval, sample.shape[:-1])
        sample_kdeval /= np.sum(sample_kdeval)

        return sample_kdeval

    def _build_mesh(self, data_fields):
        """
            Constructs a mesh array to span the full cylinder cut.

            Args:
                data_fields: list of data fields to span.
                e.g. data_fields=['vlos','Rproj'] will span the line-of-sight
                velocity in axis 0 and the projected radius in axis 1
            Returns:
                mesh: mesh array of shape self.input_shape
        """
        sample_positions = []

        for i in range(len(data_fields)):
            if data_fields[i] == 'vlos':
                sample_range = (-1.*self.vcut, self.vcut)
            elif data_fields[i] == 'Rproj':
                sample_range = (0, self.aperture)
            else:
                raise Exception('Unrecognized data field: %s' % data_fields[i])

            pos = np.linspace(sample_range[0], sample_range[1],
                              self.input_shape[i]+1)
            pos = [np.mean(pos[[i, i+1]]) for i in range(len(pos)-1)]

            sample_positions.append(pos)

        mesh = np.transpose(np.meshgrid(*sample_positions),
                            np.arange(len(data_fields), -1, step=-1))

        return mesh

    def generate_pdfs(self, catalog, data_fields, n_proc, reweightR=None):
        """
            Generates a set of normalized pdfs from a full catalog. Utilizes
            self.sample_kde and self._build_mesh . Uses multiprocessing with
            nproc processes to reduce computation time. Outputs an array of
            pdfs of shape (len(catalog), *self.input_shape).
        """
        sample = self._build_mesh(data_fields)

        data = [np.stack([catalog.gal[i][field] for field in ['vlos','Rproj']],
                         axis=1) for i in range(len(catalog))]
        
        helper = partial(self.sample_kde, 
                         sample=sample, 
                         reweightR=reweightR)

        if (n_proc is None) | (n_proc > mp.cpu_count()):
            n_proc = mp.cpu_count()


        print('running with %i processes...'%n_proc)
        with mp.Pool(processes=n_proc) as pool:
            pdfs = np.array(list(tqdm.tqdm(
                pool.imap(helper, data, chunksize=int(len(data)/(10*n_proc))),
                total=len(data))))

        pdfs = np.reshape(pdfs, (len(catalog), *(self.input_shape)))

        return pdfs

    def preprocess(self, catalog, data_fields,
                   in_train=None, in_test=None, in_val=None,
                   folds=None, n_proc=None, clean=True,
                   reweightR=None
                  ):

        # initialize
        self._update_mass_ranges(catalog)

        # subsample
        if self.sample_rate < 1:
            catalog = self._subsample(catalog, self.sample_rate)

        # assign test,train
        if in_train is None:
            in_train = self._assign_traintest(catalog, set_type=self.type_train,
                                              mass_range=self.mass_range_train)
        if in_test is None:
            in_test = self._assign_traintest(catalog, set_type=self.type_test,
                                             mass_range=self.mass_range_test)

        if in_val is None:
            in_test, in_val = self._assign_val(catalog, in_test)

        if clean:
            # clean
            keep = (in_train + in_test + in_val) > 0
            catalog = catalog[keep]
            in_train = in_train[keep]
            in_test = in_test[keep]
            in_val = in_val[keep]

        # assign folds
        if folds is None:
            folds = self._assign_folds(catalog)

        # generate pdfs
        pdfs = self.generate_pdfs(catalog, data_fields, n_proc, reweightR)

        # build output
        dtype = [('id', '<i8'), ('rotation','<i8'), ('logmass', '<f4'),
                 ('Ngal', '<i8'),
                 ('in_train', '?'), ('in_test', '?'), ('in_val', '?'),
                 ('fold', '<i4'), ('pdf', '<f4', self.input_shape)]

        out = np.zeros(shape=(len(catalog),), dtype=dtype)
        out['id'] = catalog.prop['rockstarId'].values
        out['rotation'] = catalog.prop['rotation'].values
        out['logmass'] = np.log10(catalog.prop['M200c'].values)
        out['Ngal'] = catalog.prop['Ngal']
        out['in_train'] = in_train
        out['in_test'] = in_test
        out['in_val'] = in_val
        out['fold'] = folds
        out['pdf'] = pdfs

        return out

    def augment_flip(self, X, Y, axis=1):
        X = np.append(X, np.flip(X, axis=axis), axis=0)
        Y = np.append(Y, Y, axis=0)
        return X, Y

    def save(self, fileheader, version=False, verbose=False):
        # version-ing
        if version & os.path.isfile(fileheader+'_datpar.p'):
            v = 1
            while v < 10e4:
                if os.path.isfile(fileheader+str(v)+'_datpar.p'):
                    v += 1
                else:
                    fileheader += str(v)
                    break

        # save data manager parameters
        with open(fileheader+'_datpar.p', 'wb') as f:
            pickle.dump(self.__dict__, f)
        if verbose:
            print('Saved data manager to %s' % (fileheader+'_datpar.p'))

    def load(self, fileheader, verbose=False):
        # load data manager parameters
        with open(fileheader+'_datpar.p', 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        if verbose:
            print('Loaded data manager from %s' % (fileheader+'_datpar.p'))

    def get_bin_edges(self):
        return np.linspace(*(self.bin_limits), self.nbins+1)

    def get_bin_centers(self):
        return np.convolve(self.get_bin_edges(),
                           [0.5, 0.5], mode='valid')

    def bin_to_point(self, Y):
        if self.bin_limits is None:
            raise Exception('No regularization limits set.')
        return np.dot(Y, self.get_bin_centers())

    def bin_to_variance(self, Y):
        mean = self.bin_to_point(Y)

        x = self.get_bin_centers()[np.newaxis,:].repeat(len(mean), axis=0)
        mean = mean[:,np.newaxis].repeat(self.nbins, axis=1)

        return np.sum(((x-mean)**2)*Y, axis=1)

    def bin_to_percentiles(self, Y, p):
        # FIX
        print('Warning: CDF sums to > 1!')
        Y_CDF = np.zeros(shape=(Y.shape[0], Y.shape[1]+1))
        Y_CDF[:, 0] = 0
        for i in range(1, Y.shape[-1]+1):
            Y_CDF[:, i] = Y_CDF[:, i-1] + Y[:, i-1]

        return np.array([np.interp(p, Y_CDF[i], self.get_bin_edges())
                         for i in range(len(Y_CDF))])

class RegressManager(BaseManager):
    def __init_spec__(self, reg_limits=None, bin_limits=None, nbins=None):
        self.reg_limits = reg_limits
        self.bin_limits = bin_limits
        self.nbins = nbins

    def regularize(self, Y, use_cache=True):
        if use_cache:
            if self.reg_limits is None:
                raise Exception('No regularization limits set.')
        else:
            self.reg_limits = (np.float64(Y.min()), np.float64(Y.max()))

        return ((2*Y - self.reg_limits[0] - self.reg_limits[1]) /
                (self.reg_limits[1]-self.reg_limits[0]))

    @check_reg_limits
    def deregularize(self, Y):
        return ((Y*(self.reg_limits[1]-self.reg_limits[0])) +
                self.reg_limits[0] + self.reg_limits[1])/2.
