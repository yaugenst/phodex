import autograd.numpy as np


def gray_indicator(x):
    return np.mean(4 * x * (1 - x))


def sigma_from_feature_size(feature_size: float, resolution: int) -> float:
    return (feature_size * resolution) / np.sqrt(3)
