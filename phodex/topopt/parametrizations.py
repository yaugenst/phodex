import autograd.numpy as np

from phodex.topopt.filters import gaussian_filter
from phodex.topopt.projections import sigmoid, simp


def sigmoid_parametrization(shape, sigma, alpha, beta=0.5, flat=True):
    def _parametrization(x):
        x = np.reshape(x, shape)
        x = gaussian_filter(x, sigma)
        x = sigmoid(x, alpha, beta)
        return x.ravel() if flat else x

    return _parametrization


def simp_parametrization(shape, sigma, vmin, vmax, penalty=3.0, flat=True):
    def _parametrization(x):
        x = np.reshape(x, shape)
        x = gaussian_filter(x, sigma)
        x = simp(x, vmin, vmax, penalty)
        return x.ravel() if flat else x

    return _parametrization
