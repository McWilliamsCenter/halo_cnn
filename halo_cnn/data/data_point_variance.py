

import numpy as np
from functools import partial
import tqdm
import multiprocessing as mp
from scipy.stats import gaussian_kde
from scipy import linalg

from .data_point import PointManager


GaussManager = PointManager


class GaussDropoutManager(GaussManager):
    def get_mean(self, Y):
        return self.deregularize(np.mean(Y[..., 0], axis=-1))

    def get_variance(self, Y):
        return self.deregularize_var(np.var(Y[..., 0], axis=-1) +
                                     np.mean(Y[..., 1], axis=-1))


class GaussDropoutManagerOBS(GaussDropoutManager):
    def load_catalog(self, file, cache=True):
        """Loads a catalog file from disk and saves vcut,aperture settings"""
        if file is str:
            catalog = Catalog().load(file)
        else:
            catalog = file

        if cache:
            self.vcut = catalog.par['vcut']
            self.aperture = catalog.par['aperture']

        return catalog

    def sample_kde(self, data, sample,  data_fields, rwfunc=None):
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
        
        # reweight
        if rwfunc is not None:
            weights = rwfunc(data)
        else:
            weights = None
        
        # generate kde object
        data = np.stack([data[field] for field in  data_fields], axis=1).T
        kde = gaussian_kde(data, self.bandwidth, weights)

        # sample kdes
        sample_flat = np.transpose(sample, (len(sample.shape)-1,
                                            *np.arange(len(sample.shape)-1)))
        sample_flat = np.reshape(sample_flat,
                                 (sample.shape[-1],
                                  np.product(sample.shape[:-1])))

        sample_kdeval = kde.evaluate(sample_flat)

        sample_kdeval = np.reshape(sample_kdeval, sample.shape[:-1])
        sample_kdeval /= np.sum(sample_kdeval)

        return sample_kdeval
    
    
    
    def generate_pdfs(self, catalog, data_fields, n_proc, rwfunc=None):
        """
            Generates a set of normalized pdfs from a full catalog. Utilizes
            self.sample_kde and self._build_mesh . Uses multiprocessing with
            nproc processes to reduce computation time. Outputs an array of
            pdfs of shape (len(catalog), *self.input_shape).
        """
        sample = self._build_mesh(data_fields)
        
        helper = partial(self.sample_kde, 
                         sample=sample, 
                         rwfunc=rwfunc,
                         data_fields=data_fields)

        if (n_proc is None) | (n_proc > mp.cpu_count()):
            n_proc = mp.cpu_count()

        if n_proc == 1:
            pdfs = np.array(list(map(helper, tqdm.tqdm(catalog.gal))))
        else:
            print(f'running with {n_proc} processes...')
            with mp.Pool(processes=n_proc) as pool:
                pdfs = np.array(list(tqdm.tqdm(
                    pool.imap(helper, catalog.gal, chunksize=int(len(catalog)/(10*n_proc))),
                    total=len(catalog))))

        pdfs = np.reshape(pdfs, (len(catalog), *(self.input_shape)))

        return pdfs

    def preprocess(self, catalog, data_fields,
                   in_train=None, in_test=None, in_val=None,
                   folds=None, n_proc=None, clean=True,
                   rwfunc=None
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
        pdfs = self.generate_pdfs(catalog, data_fields, n_proc, rwfunc)

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


from scipy.stats import norm

class GaussDropoutManagerOBS_cov(GaussDropoutManagerOBS):
    def __init_spec__(self, unc_vlos=3e2, unc_Rproj=0.1, reg_limits=None, bin_limits=None, nbins=None):
        self.reg_limits = reg_limits
        self.bin_limits = bin_limits
        self.nbins = nbins
        
        self.unc_vlos = unc_vlos
        self.unc_Rproj = unc_Rproj
        
    def sample_kde(self, data, sample,  data_fields, rwfunc=None):
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
        
        # reweight
        if rwfunc is not None:
            weights = rwfunc(data)
        else:
            weights = None
        
        # generate kde object
        data = np.stack([data[field] for field in  data_fields], axis=1).T
        kde = gaussian_kde(data, self.bandwidth, weights=weights)
        kde.covariance = np.diag([self.unc_vlos**2, self.unc_Rproj**2])
        kde.inv_cov = linalg.inv(kde.covariance)

        # sample kdes
        sample_flat = np.transpose(sample, (len(sample.shape)-1,
                                            *np.arange(len(sample.shape)-1)))
        sample_flat = np.reshape(sample_flat,
                                 (sample.shape[-1],
                                  np.product(sample.shape[:-1])))
        
        

        sample_kdeval = kde.evaluate(sample_flat)

        sample_kdeval = np.reshape(sample_kdeval, sample.shape[:-1])
        sample_kdeval /= np.sum(sample_kdeval)

        return sample_kdeval
