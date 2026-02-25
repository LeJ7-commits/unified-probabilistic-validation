import numpy as np

def normalize_nonnegative_weights(w, eps=1e-12):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 0.0)
    s = np.sum(w)
    if s < eps:
        return np.ones_like(w)
    return w / s