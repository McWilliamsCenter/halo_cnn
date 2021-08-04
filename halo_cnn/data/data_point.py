
from scipy.special import logsumexp
from scipy.stats import norm

from .data_base import np, gaussian_kde, RegressManager, check_reg_limits


class PointManager(RegressManager):

    def get_percentiles(self, mean, var, p):
        return np.array([norm.ppf(i, mean, np.sqrt(var)) for i in p]).T

    @check_reg_limits
    def deregularize_var(self, var):
        return var/((2/np.diff(self.reg_limits))**2)

    def get_pdf(self, mean, var, x):
        return np.array([norm.pdf(j, mean, np.sqrt(var)) for j in x]).T
    
    def get_cdf(self, mean, var, x):
        return np.array([norm.cdf(j, mean, np.sqrt(var)) for j in x]).T

    def reweight_by_prior(self, mean, var, N, prior_dist):
        prior_range = (min(prior_dist), max(prior_dist))
        self.bin_limits = prior_range
        self.nbins = N

        Y = self.get_pdf(mean, var, self.get_bin_centers()).T

        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, 0)

        kde = gaussian_kde(prior_dist, bw_method=0.3)
        kde_pdf = kde(self.get_bin_centers())

        Y /= kde_pdf[np.newaxis, :]
        Y /= np.sum(Y, axis=-1)[:, np.newaxis]

        if Y.shape[0] == 1:
            Y = Y[0, ...]

        return Y


class PointDropoutManager(PointManager):
    def get_mean(self, Y):
        return self.deregularize(np.mean(Y, axis=-1))

    def get_variance(self, Y, tau):
        return self.deregularize_var(np.var(Y, axis=-1) + 1./(tau))

    def get_ll(self, Y_hat, Y, tau):
        ll = logsumexp(-0.5*tau*(Y_hat - Y.reshape(-1, 1))**2, axis=1) - \
            np.log(Y_hat.shape[1]) - 0.5*np.log(2*np.pi) + 0.5 * np.log(tau)
        return np.mean(ll)
