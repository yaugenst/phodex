import autograd.numpy as np


def gray_indicator(x):
    return np.mean(4 * x * (1 - x))
