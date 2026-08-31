"""
Fisher information matrix for the SINDy coefficient estimates, used by
the dual iLQG curiosity term.

Ported from Katie's CODES/KATIE'S CODE/.../compute_fisher_information.m.

For the linear-Gaussian regression model this pipeline assumes
(dX = Theta @ Xi + eps, eps ~ N(0, sigma^2 I)), the Fisher information
matrix has a closed form:

    F = (1 / sigma^2) * Theta_n.T @ Theta_n   (Theta_n = column-normalised Theta)

Its inverse is the Cramer-Rao lower bound: the theoretical floor on how
uncertain any estimator's Xi can be, given this exact Theta and noise
level. F is not learned from data -- it's a property of the DESIGN
(which library terms were evaluated, how much variation the training
data gave them). The dual iLQG controller uses it to quantify expected
information gain from a proposed action: curiosity means preferring
actions expected to shrink parameter uncertainty fastest, which
requires a measure of CURRENT uncertainty -- exactly what F (via its
inverse) provides. This is the connective tissue between the batch
SINDy identification pipeline (build_sindy_library.py through
validate_sindy_model.py) and the online dual-control loop
(todorov_estimator.py / spkf_function.py), which carries a parameter
covariance around during real-time operation.

INPUTS
  Theta      (N, n_terms)  library matrix (physical units -- normalised
             internally, same convention as stlsq.py/prune_sindy_bic.py)
  noise_std  scalar  assumed observation noise standard deviation

OUTPUT
  F  (n_terms, n_terms)  Fisher information matrix (normalised-
     coefficient space), guaranteed symmetric

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - Same ddof=1 (N-1 std) correctness trap as every other file in this
    pipeline: `std(Theta,0,1)` needs ddof=1, not NumPy's default ddof=0.
  - MATLAB's `warning(...)` calls (near-zero noise_std fallback; non-PSD
    F regularisation) are translated to Python's `warnings.warn` rather
    than dropped -- unlike the dead/unreachable warnings dropped
    elsewhere in this port (e.g. simulate_system.py's isreal check),
    both of these ARE reachable with realistic inputs and are meant to
    surface as actionable diagnostics, not silently vanish.
  - `eig(F)` (MATLAB) -> `np.linalg.eigvalsh(F)`: F is symmetrised
    immediately before this check, so eigvalsh (which assumes/exploits
    symmetry) is the natural NumPy equivalent, not a behavioural
    deviation. Theta_n.T @ Theta_n is mathematically guaranteed PSD in
    exact arithmetic; a negative eigenvalue here can only be
    floating-point roundoff on a near-singular Theta, never a sign that
    something is conceptually wrong -- the regularisation branch is a
    numerical safety net. It's not something a hand-built test input can
    reliably force, but it fires naturally (and is asserted on) in
    test_compute_fisher_information.py's real-dims composition test:
    with poly_order=2+cross, build_sindy_library.py's Theta has MORE
    candidate terms than there are training observations, which makes
    Theta.T @ Theta genuinely rank-deficient (not just poorly
    conditioned) -- exactly the situation this branch exists for.
"""

import warnings

import numpy as np


def compute_fisher_information(Theta, noise_std):
    sigma2 = noise_std ** 2

    if sigma2 < np.finfo(float).eps:
        sigma2 = 1e-6
        warnings.warn("compute_fisher_information: noise_std near zero. "
                        "Using 1e-6.")

    # normalise Theta columns so F is expressed in normalised-
    # coefficient space, consistent with stlsq.py / prune_sindy_bic.py
    col_std = Theta.std(axis=0, ddof=1)
    col_std[col_std < 1e-10] = 1.0
    Theta_n = Theta / col_std

    F = (1.0 / sigma2) * (Theta_n.T @ Theta_n)
    F = (F + F.T) / 2.0  # correct floating-point asymmetry

    eigvals = np.linalg.eigvalsh(F)
    if np.any(eigvals < 0):
        warnings.warn("compute_fisher_information: F is not positive "
                        "semi-definite. Adding regularisation.")
        F = F + 1e-8 * np.eye(F.shape[0])

    return F
