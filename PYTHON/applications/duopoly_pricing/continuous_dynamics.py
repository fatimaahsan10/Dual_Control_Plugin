"""
Duopoly-pricing plant dynamics: continuous-time demand relaxation toward
its price-implied level, for TWO sellers whose demands depend on BOTH
prices, under unknown own- and cross-price sensitivities. Direct 2-seller
generalization of applications/dynamic_pricing/continuous_dynamics.py.

FRAMING (see 2026-08-26 math-review discussion): this models ONE
controller jointly setting both sellers' prices to maximize COMBINED
profit (e.g. a platform or multi-brand retailer) -- not two
independently-optimizing competitive agents. "Competition" shows up only
as cross-price terms coupling the two demand curves, not as a
game-theoretic equilibrium (this codebase implements optimal control,
not game theory).

STATE VECTOR (nx = 2): x1 = D_A (seller A realized demand), x2 = D_B
(seller B realized demand). Kept minimal, same convention as
dynamic_pricing/continuous_dynamics.py.

ACTION VECTOR (nu = 2): u1 = p_A (seller A's price), u2 = p_B (seller
B's price).

AUGMENTED PARAMETERS (augment_states=True): xa[2:6] = [b_AA, b_AB, b_BA,
b_BB] -- FIXED ORDERING CONVENTION every plant file in this package
agrees on:
    b_AA : seller A's own-price sensitivity
    b_AB : seller A's cross-price sensitivity to seller B's price
    b_BA : seller B's cross-price sensitivity to seller A's price
    b_BB : seller B's own-price sensitivity
Curiosity here is genuinely 2-D: b_AA/b_BB are informed by each seller's
OWN price (dD_A/db_AA = -p_A), while b_AB/b_BA are informed by the OTHER
price (dD_A/db_AB = +p_B) -- since one controller sets and observes both,
neither cross-term suffers a multi-agent identifiability problem, but
learning all four still requires the controller to move BOTH prices, not
just its own profit-maximizing one.

DEMAND MODEL (linear, substitute-goods sign convention -- cross-price
coefficients POSITIVE: a higher rival price pushes customers toward you;
flip sign for complements):
    D_A(p_A, p_B; b) = a_A - b_AA*p_A + b_AB*p_B
    D_B(p_A, p_B; b) = a_B - b_BB*p_B + b_BA*p_A

DYNAMICS (partial adjustment, per seller):
    xdot_1 = eta_A * (D_A(p_A, p_B; b) - x1)
    xdot_2 = eta_B * (D_B(p_A, p_B; b) - x2)
    bdot   = 0   (zero-drift random walk under estimation, same
                  convention as dynamic_pricing/continuous_dynamics.py)

GUARDRAIL: none applied here -- the floor clamp x>=0 lives in
dynamics.py's discrete step (continuous-time derivatives are left
unclamped), same split as dynamic_pricing/continuous_dynamics.py.
"""

import numpy as np


def continuous_dynamics(dt, xa, u, constants, augment_states, u_lims,
                          u_lim_method):
    """
    Returns
    -------
    xdot : (nx [+ n_p], K)   state (and, if augmented, parameter) derivatives
    nx   : 2
    n_p  : 4 if augment_states else 0
    F    : (nx, nx)   dynamics noise scale matrix
    Fp   : (n_p, n_p) parameter noise scale matrix, empty if not augmented
    """
    nx = 2
    K = xa.shape[1]

    if u.shape[1] == 1 and K > 1:
        u = np.tile(u, (1, K))

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
              + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)

    a_A, a_B = constants.a_A, constants.a_B
    eta_A, eta_B = constants.eta_A, constants.eta_B

    x_A, x_B = xa[0, :], xa[1, :]
    p_A, p_B = u[0, :], u[1, :]

    if augment_states:
        b_AA, b_AB, b_BA, b_BB = xa[2, :], xa[3, :], xa[4, :], xa[5, :]
        n_p = 4
    else:
        b_AA = np.full(K, constants.b_prior[0])
        b_AB = np.full(K, constants.b_prior[1])
        b_BA = np.full(K, constants.b_prior[2])
        b_BB = np.full(K, constants.b_prior[3])
        n_p = 0

    D_A = a_A - b_AA * p_A + b_AB * p_B
    D_B = a_B - b_BB * p_B + b_BA * p_A

    f_A = (eta_A * (D_A - x_A)).reshape(1, K)
    f_B = (eta_B * (D_B - x_B)).reshape(1, K)
    f = np.vstack([f_A, f_B])

    if augment_states:
        fp = np.zeros((n_p, K))  # parameter dynamics: zero-mean random walk
        xdot = np.vstack([f, fp])
    else:
        xdot = f

    F = np.diag([constants.F_dyn_A, constants.F_dyn_B])
    Fp = (np.diag([constants.F_param_A, constants.F_param_A,
                     constants.F_param_B, constants.F_param_B])
           if augment_states else np.zeros((0, 0)))

    return xdot, nx, n_p, F, Fp
