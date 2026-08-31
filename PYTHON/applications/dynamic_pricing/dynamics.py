"""
Euler-integrates continuous_dynamics.py's xdot = f(x, u; b) into a
one-step-ahead discrete map, with dynamics noise added -- same structure
as applications/sci_rehab/discrete_state_dynamics.py wrapping
continuous_state_dynamics.py.

    xa_{k+1} = xa_k + dt * f(xa_k, u_k) + F_a @ w * sqrt(dt)

f is continuous_dynamics.py's plant; F_a = blkdiag(F, Fp) is the (state,
parameter) noise-diffusion matrix it returns -- constant here, not a
function of x/u/k, so the sqrt(dt)-scaled Euler-Maruyama noise term is
exactly F_a @ w * sqrt(dt) (same convention as
extensions/dual_control/forward_pass.py's c/Cx/Cu noise terms).

Matches DynamicsFn's contract exactly (extensions/dual_control/protocols.py):
    dynamics(dt, xa, u, constants, w, noise_index, augment_states,
             w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method)
        -> (nxa, K) array
noise_index=False/0 -> deterministic-plus-noise next state.
noise_index=i (1-indexed) -> the i-th column of the noise-diffusion
matrix F_a, broadcast across all K columns (F_a doesn't depend on x/u
here) -- this is what forward_pass.py differentiates to build Cx/Cu.

GUARDRAIL: realized demand is clamped to x >= 0 after the step -- linear
demand can go negative for extreme price/noise combinations (flagged,
known validity-range limitation of the linear demand curve, same
precedent as continuous_state_dynamics.py's SCIM/BBS floor/ceiling
clamp).

`dyn_noise_reg_lambda` is accepted for signature fidelity with the
shared DynamicsFn contract but unused here (this plant's noise scale is
constant, not iteration-regularized) -- same convention as
discrete_state_dynamics.py's unused `lam`.
"""

import numpy as np

from applications.dynamic_pricing.continuous_dynamics import continuous_dynamics


def dynamics(dt, xa, u, constants, w, noise_index, augment_states,
              w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method):
    Kplus1 = xa.shape[1]

    if w.shape[1] == 1 and Kplus1 > 1:
        w = np.tile(w, (1, Kplus1))
    if u.shape[1] == 1 and Kplus1 > 1:
        u = np.tile(u, (1, Kplus1))

    fa, nx, n_p, F, Fp = continuous_dynamics(
        dt, xa, u, constants, augment_states, u_lims, u_lim_method)

    F_a = _block_diag(F, Fp)  # (nxa, nxa), constant across columns

    if noise_index:  # 1-indexed request for the noise-diffusion column
        col = F_a[:, noise_index - 1]
        return np.tile(col[:, None], (1, Kplus1))

    if not w_from_filter:
        noise = (F_a @ w) * np.sqrt(dt)
    else:
        noise = w

    x_next = xa + fa * dt + noise
    x_next[0, :] = np.maximum(x_next[0, :], 0.0)  # demand floor guardrail
    return x_next


def _block_diag(*mats):
    mats = [np.atleast_2d(m) for m in mats]
    total_rows = sum(m.shape[0] for m in mats)
    total_cols = sum(m.shape[1] for m in mats)
    out = np.zeros((total_rows, total_cols))
    i = j = 0
    for m in mats:
        r, c = m.shape
        out[i:i + r, j:j + c] = m
        i += r
        j += c
    return out
