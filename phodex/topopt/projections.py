import autograd.numpy as np


def ramp(x, w=1e-2, b=0.5):
    return (x > (b - w)) * x + (x > (b + w)) * (1 - x)


def sigmoid(x, a, b=0.5):
    if a == 0:
        return x
    num = np.tanh(a * b) + np.tanh(a * (x - b))
    denom = np.tanh(a * b) + np.tanh(a * (1 - b))
    return num / denom


def simp(x, vmin, vmax, penalty=3.0):
    return vmin + x**penalty * (vmin - vmax)
