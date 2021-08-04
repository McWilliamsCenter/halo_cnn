
from .data_base import np, gaussian_kde, BaseManager


class ClassifyManager(BaseManager):

    def __init_spec__(self, bin_limits, nbins):
        self.bin_limits = bin_limits
        self.nbins = nbins

    def point_to_bin(self, Y):
        out = np.zeros((len(Y), self.nbins))
        out[np.arange(len(Y)),
            ((Y-self.bin_limits[0])*self.nbins /
             (self.bin_limits[1]-self.bin_limits[0])).astype(int)] = 1
        return out

    def reweight_by_prior(self, Y, prior_dist):
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, 0)

        kde = gaussian_kde(prior_dist, bw_method=0.3)
        kde_pdf = kde(self.get_bin_centers())
        kde_pdf[:np.argmax(self.get_bin_centers() > np.log10(self.mass_range_train[0]))] = np.inf
        kde_pdf[np.argmin(self.get_bin_centers() < np.log10(self.mass_range_train[1])):] = np.inf

        Y /= kde_pdf[np.newaxis, :]
        Y /= np.sum(Y, axis=-1)[:, np.newaxis]

        if Y.shape[0] == 1:
            Y = Y[0, ...]

        return Y

class ClassifyDropoutManager(ClassifyManager):
    def softmax(self, Y):
        u = np.exp(Y)
        return u/np.sum(u, axis=-1)[:, np.newaxis]

    def bin_mean(self, Y):
        if self.bin_limits is None:
            raise Exception('No regularization limits set.')
        return self.softmax(np.mean(Y, axis=-2))
