import numpy as np


# FIXME: Docstring incomplete
def drag(wf):
    return np.gradient(wf)


# FIXME: Docstring incomplete
def gaussian(len, sigma):
    xs = np.arange(len) - len//2
    ys = np.exp(-xs**2/(2*sigma**2))
    return ys


# FIXME: Docstring incomplete
def smoothed_constant(len, edge):
    xs = np.arange(len)
    ys = np.ones_like(xs, dtype=float)

    w = np.pi / edge / 2.
    ys[:edge] *= 1 - np.cos(w * np.arange(edge)) ** 2
    ys[-edge:] *= (1 - np.cos(w * np.arange(edge)) ** 2)[::-1]

    return ys
