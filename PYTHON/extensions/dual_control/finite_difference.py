"""
Vectorized forward-difference Jacobian.

Ported from Andrew's CODES/ANDREW'S CODE/finite_difference.m (originally by
Emo Todorov). Used throughout Andrew's forward_pass.m to linearize
dynamics/measurement/cost since (unlike Todorov's demo_linear.m) they're
not differentiated analytically.
"""

import numpy as np


def finite_difference(fun, x, h=2 ** -17):
    """
    Parameters
    ----------
    fun : callable, vectorized over columns: (n, M) -> (m, M)
    x   : (n, K)
    h   : perturbation size

    Returns
    -------
    J : (m, n, K)   J[:, i, k] = (fun(x[:,k] + h*e_i) - fun(x[:,k])) / h
    """
    n, K = x.shape

    # one big vectorized call: column block 0 is the baseline x, block i
    # (i=1..n) is x with dimension i-1 perturbed by h, for every one of the
    # K batch columns at once.
    X = np.empty((n, (n + 1) * K))
    X[:, :K] = x
    for i in range(n):
        X[:, (i + 1) * K:(i + 2) * K] = x + (h * np.eye(n)[:, i])[:, None]

    Y = fun(X)
    m = Y.shape[0]
    Y = Y.reshape(m, n + 1, K)

    J = (Y[:, 1:, :] - Y[:, 0:1, :]) / h
    return J
