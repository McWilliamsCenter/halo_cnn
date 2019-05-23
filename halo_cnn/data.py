# This file contains the manager for loading and preprocessing
# halo_cnn data

import numpy as np
import multiprocessing as mp
import pandas as pd
import tqdm

from scipy.stats import gaussian_kde
from functools import partial


# FUNCTIONS

from tools.catalog import Catalog


class HaloCNNDataManager():
    def __init__(self, input_shape,
                 bandwidth=0.25, sample_rate=1.0, nfolds=10,
                 mass_range_train=(None, None), mass_range_test=(None, None),
                 dn_dlogm=10.**-5.2, dlogm=0.02,
                 vcut=None, aperture=None):
        # initialization
        self.input_shape = input_shape
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate
        self.nfolds = nfolds
        self.mass_range_train = mass_range_train
        self.mass_range_test = mass_range_test
        self.dn_dlogm = dn_dlogm
        self.dlogm = dlogm
        self.vcut = vcut
        self.aperture = aperture

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

        return

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
        n_per_bin = int(self.dn_dlogm*1000**3*self.dlogm)

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
            in_array = self._assign_all(catalog, mass_range, 3)
        else:
            raise Exception('Unknown traintest type:' + str(set_type))

        return in_array

    def _assign_folds(self, catalog):
        """Use rank-ordering to assign folds evenly for all masses"""
        ids_sorted = catalog.prop[['rockstarId', 'M200c']].drop_duplicates()
        ids_sorted = ids_sorted.sort_values(['M200c'])['rockstarId']
        fold_ind = pd.Series(np.arange(len(ids_sorted)) % self.nfolds,
                             index=ids_sorted)
        folds = fold_ind[catalog.prop['rockstarId']].values

        return folds

    def sample_kde(self, data, sample):
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
        if data.shape[-1] != sample.shape[-1]:
            raise Exception('Data of dimension %(data_dim)s does not align'
                            ' with sample dimension %(samp_dim)s' %
                            {'data_dim': data.shape[-1],
                             'samp_dim': sample.shape[-1]})

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
                sample_range = (-self.vcut, self.vcut)
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

    def generate_pdfs(self, catalog, data_fields, n_proc):
        """
            Generates a set of normalized pdfs from a full catalog. Utilizes
            self.sample_kde and self._build_mesh . Uses multiprocessing with
            nproc processes to reduce computation time. Outputs an array of
            pdfs of shape (len(catalog), *self.input_shape).
        """
        sample = self._build_mesh(data_fields)

        data = [np.stack([catalog.gal[i][field] for field in data_fields],
                         axis=1) for i in range(len(catalog))]

        helper = partial(self.sample_kde, sample=sample)

        if n_proc is None:
            n_proc = mp.cpu_count()

        with mp.Pool(processes=n_proc) as pool:
            pdfs = np.array(list(tqdm.tqdm(
                pool.imap(helper, data, chunksize=int(len(data)/(10*n_proc))),
                total=len(data))))

        return pdfs

    def preprocess(self, catalog, data_fields,
                   in_train=None, in_test=None,
                   folds=None, n_proc=None):

        # initialize
        self._update_mass_ranges(catalog)

        # subsample
        if self.sample_rate < 1:
            catalog = self._subsample(catalog, self.sample_rate)

        # assign test,train
        if in_train is None:
            in_train = self._assign_traintest(catalog, set_type='flat',
                                              mass_range=self.mass_range_train)
        if in_test is None:
            in_test = self._assign_traintest(catalog, set_type='all',
                                             mass_range=self.mass_range_test)

        # clean
        keep = (in_train + in_test) > 0
        catalog = catalog[keep]
        in_train = in_train[keep]
        in_test = in_test[keep]

        # assign folds
        if folds is None:
            folds = self._assign_folds(catalog)

        # generate pdfs
        pdfs = self.generate_pdfs(catalog, data_fields, n_proc)

        # build output
        dtype = [('id', '<i8'), ('logmass', '<f4'), ('Ngal', '<i8'),
                 ('in_train', '?'), ('in_test', '?'),
                 ('fold', '<i4'), ('pdf', '<f4', self.input_shape)]

        out = np.zeros(shape=(len(catalog),), dtype=dtype)
        out['id'] = catalog.prop['rockstarId'].values
        out['logmass'] = np.log10(catalog.prop['M200c'].values)
        out['Ngal'] = catalog.prop['Ngal']
        out['in_train'] = in_train
        out['in_test'] = in_test
        out['fold'] = folds
        out['pdf'] = pdfs

        return out

    def regularize(self, Y, use_cache=False):
        if use_cache:
            if self.reg_limits is None:
                raise Exception('No regularization limits set.')
        else:
            self.reg_limits = (Y.min(), Y.max())

        return ((2*Y - self.reg_limits[0] - self.reg_limits[1]) /
                (self.reg_limits[1]-self.reg_limits[0]))

    def deregularize(self, Y):
        if self.reg_limits is None:
            raise Exception('No regularization limits set.')

        return ((Y*(self.reg_limits[1]-self.reg_limits[0])) +
                self.reg_limits[0] + self.reg_limits[1])/2.

    def augment_flip(self, X, Y, axis=1):
        X = np.append(X, np.flip(X, axis=axis), axis=0)
        Y = np.append(Y, Y, axis=0)
        return X, Y

    def save(self, filename):
        return

    def load(self, filename):
        return
