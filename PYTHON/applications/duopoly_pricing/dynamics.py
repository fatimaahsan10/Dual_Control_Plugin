"""
Euler-integrates continuous_dynamics.py's xdot = f(x, u; b) into a
one-step-ahead discrete map, with dynamics noise added -- direct 2-seller
generalization of applications/dynamic_pricing/dynamics.py (same
structure, same _block_diag helper, same Euler-Maruyama convention).

    xa_{k+1} = xa_k + dt * f(xa_k, u_k) + F_a @ w * sqrt(dt)

Matches DynamicsFn's contract exactly (extensions/dual_control/protocols.py):
    dynamics(dt, xa, u, constants, w, noise_index, augment_states,
             w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method)
        -> (nxa, K) array
noise_index=False/0 -> deterministic-plus-noise next state.
noise_index=i (1-indexed) -> the i-th column of the noise-diffusion
matrix F_a, broadcast across all K columns (F_a doesn't depend on x/u
here) -- this is what forward_pass.py differentiates to build Cx/Cu.

GUARDRAIL: both realized demands are clamped to x >= 0 after the step --
same known validity-range limitation of the linear demand curve as
dynamic_pricing/dynamics.py, now applied to both sellers. Worst-case
corner for this clamp binding is one seller's OWN price near its ceiling
while the RIVAL's price is near its floor (minimal offsetting cross-price
boost) -- see run_duopoly_demo.py's price-bound choice.

`dyn_noise_reg_lambda` is accepted for signature fidelity with the shared
DynamicsFn contract but unused here, same convention as
dynamic_pricing/dynamics.py's unused `dyn_noise_reg_lambda`.
"""

import numpy as np

from applications.duopoly_pricing.continuous_dynamics import continuous_dynamics


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
    x_next[0, :] = np.maximum(x_next[0, :], 0.0)  # demand floor guardrail, seller A
    x_next[1, :] = np.maximum(x_next[1, :], 0.0)  # demand floor guardrail, seller B
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
