"""
Force a symmetric matrix to be positive-definite with eigenvalues >= epsilon.

Ported from Andrew's CODES/ANDREW'S CODE/makePD.m ("from Todorov 2007, p
1445"). Used by SPKF_function.m to clean up the state covariance after the
sigma-point update. Same two strategies already inlined in backward_pass.py
for reg_type 3 (method 1) and reg_type 4 (method 2).

DEVIATION FROM THE LITERAL MATLAB SOURCE: an unrecognized method raises
ValueError instead of printing a message and returning a NaN-filled matrix
-- silently propagating NaNs is a worse failure mode than failing loudly.
"""

import numpy as np


def make_pd(H, epsilon, method):
    if method == 1:
        # uniform shift: every eigenvalue moves by the same amount so the
        # smallest lands exactly on epsilon
        lambda_min = np.linalg.eigvalsh(H).min()
        return H + (epsilon - lambda_min) * np.eye(H.shape[0])

    if method == 2:
        # clip only the eigenvalues below epsilon, leave the rest alone
        eigval, eigvec = np.linalg.eigh(H)
        eigval = np.maximum(eigval, epsilon)
        return eigvec @ np.diag(eigval) @ eigvec.T

    raise ValueError(f"method must be 1 or 2, got {method!r}")
