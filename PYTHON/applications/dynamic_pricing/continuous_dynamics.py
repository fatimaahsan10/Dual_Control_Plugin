"""
Dynamic-pricing plant dynamics: continuous-time demand relaxation toward
its price-implied level, under an unknown price sensitivity b.

STATE VECTOR (nx = 1): x1 = realized demand (units sold this period).
Only x1 evolves; there are no static carried-along attributes here
(unlike Katie's SCI plant), since this test case was deliberately kept
minimal (see CLAUDE.md's dynamic-pricing math review, flagged
simplification list).

ACTION VECTOR (nu = 1): u1 = price charged this period.

AUGMENTED PARAMETER (augment_states=True): xa[1] = b_hat, the online
estimate of price sensitivity b (the slope of the linear demand curve
D(u;b) = a - b*u). Curiosity drives price experimentation that reduces
uncertainty about b, not just actions that maximize profit under the
current point estimate -- same role as continuous_state_dynamics.py's
augmented SINDy coefficients.

DEMAND MODEL: D(u;b) = a - b*u (linear demand curve; b is the "price
sensitivity" slope, not elasticity in the strict %-change sense -- see
CLAUDE.md's terminology note from the math review).

DYNAMICS (partial adjustment -- demand relaxes toward its price-implied
level rather than jumping instantly):
    xdot_1 = eta * (a - b*u - x1)
    xdot_2 = 0   (b: zero-drift random walk under estimation, same
                  convention as continuous_state_dynamics.py's `fp`)

GUARDRAIL: none applied here (the floor clamp x>=0 lives in
dynamics.py's discrete step, mirroring continuous_state_dynamics.py /
discrete_state_dynamics.py's own split -- continuous-time derivatives
are left unclamped; the clamp is a property of the discrete update).
"""

import numpy as np


def continuous_dynamics(dt, xa, u, constants, augment_states, u_lims,
                          u_lim_method):
    """
    Returns
    -------
    xdot : (nx [+ n_p], K)   state (and, if augmented, parameter) derivatives
    nx   : 1
    n_p  : 1 if augment_states else 0
    F    : (nx, nx)   dynamics noise scale matrix
    Fp   : (n_p, n_p) parameter noise scale matrix, empty if not augmented
    """
    nx = 1
    K = xa.shape[1]

    if u.shape[1] == 1 and K > 1:
        u = np.tile(u, (1, K))

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
              + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)

    a, eta = constants.a, constants.eta
    x = xa[0, :]
    price = u[0, :]

    if augment_states:
        b = xa[1, :]
        n_p = 1
    else:
        b = np.full(K, constants.b_prior)
        n_p = 0

    f = (eta * (a - b * price - x)).reshape(1, K)

    if augment_states:
        fp = np.zeros((n_p, K))  # parameter dynamics: zero-mean random walk
        xdot = np.vstack([f, fp])
    else:
        xdot = f

    F = np.array([[constants.F_dyn]])
    Fp = np.array([[constants.F_param]]) if augment_states else np.zeros((0, 0))

    return xdot, nx, n_p, F, Fp
