

import numpy as np
from .data_point import PointManager


GaussManager = PointManager


class GaussDropoutManager(GaussManager):
    def get_mean(self, Y):
        return self.deregularize(np.mean(Y[..., 0], axis=-1))

    def get_variance(self, Y):
        return self.deregularize_var(np.var(Y[..., 0], axis=-1) +
                                     np.mean(Y[..., 1], axis=-1))
