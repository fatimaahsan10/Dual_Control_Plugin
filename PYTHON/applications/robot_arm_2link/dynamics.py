"""
Euler-integrates continuous_dynamics.py's xdot = f(x, u; m2) into a
one-step-ahead discrete map, with dynamics noise added -- same structure
as applications/duopoly_pricing/dynamics.py:

    xa_{k+1} = xa_k + dt * f(xa_k, u_k) + F_a @ w * sqrt(dt)

Matches DynamicsFn's contract exactly (extensions/dual_control/protocols.py):
    dynamics(dt, xa, u, constants, w, noise_index, augment_states,
             w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method)
        -> (nxa, K) array
noise_index=False/0 -> deterministic-plus-noise next state.
noise_index=i (1-indexed) -> the i-th column of the noise-diffusion
matrix F_a, broadcast across all K columns (F_a doesn't depend on x/u
here) -- this is what forward_pass.py differentiates to build Cx/Cu.

GUARDRAIL: when augmented, m2 (row 4) is floored at constants.m2_floor
after the Euler step -- a physical mass can't go negative, but the
random-walk parameter dynamics (fp = 0, noise added by F_a) have no
built-in floor of their own. Same "clamp the physically-invalid corner,
document it, leave everything else smooth" convention as
applications/duopoly_pricing/dynamics.py's x >= 0 demand floor. No
clamp on q1/q2/w1/w2 -- see continuous_dynamics.py's "NO ANGLE WRAPPING"
note.

`dyn_noise_reg_lambda` is accepted for signature fidelity with the
shared DynamicsFn contract but unused here, same convention as
duopoly_pricing/dynamics.py's unused `dyn_noise_reg_lambda`.
"""

import numpy as np

from applications.robot_arm_2link.continuous_dynamics import continuous_dynamics


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
    if augment_states:
        x_next[4, :] = np.maximum(x_next[4, :], constants.m2_floor)  # mass positivity guardrail
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
