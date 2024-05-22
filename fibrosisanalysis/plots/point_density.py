import numpy as np
from scipy import stats


class PointDensity:
    def __init__(self):
        pass

    @staticmethod
    def gaussian_kde(x, y):
        xy = np.vstack([x, y])
        density = stats.gaussian_kde(xy)(xy)
        return density

    @staticmethod
    def sort_by_density(x, y, index=False):
        density = PointDensity.gaussian_kde(x, y)
        idx = density.argsort()
        if index:
            return idx
        return x[idx], y[idx], density[idx]
