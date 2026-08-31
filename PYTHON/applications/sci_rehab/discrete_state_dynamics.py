"""
Euler-integrates continuous_state_dynamics.py's ẋ = f(x, u) into a
one-step-ahead discrete map, with dynamics noise added.

Ported from Katie's CODES/KATIE'S CODE/.../DiscreteStateDynamics.m.

    x_{k+1} = x_k + dt * f(x_k, u_k) + F_a @ w * sqrt(dt)

f is continuous_state_dynamics.py's SINDy-identified plant. F_a =
blkdiag(F, Fp) is the (state, parameter) noise-diffusion matrix it
returns -- constant here, not a function of x/u/k (F = F_dyn*I,
Fp = F_param*I) -- so the sqrt(dt)-scaled Euler-Maruyama noise term is
exactly F_a @ w * sqrt(dt) (same convention as Andrew's forward_pass.py
c/Cx/Cu noise terms).

WIRING NOTE: F_a is (nx + n_p, nx + n_p) -- one noise channel per
augmented state dimension. Whoever wires this into forward_pass() must
pass nw = nx + n_p, not an independently-chosen channel count like
Andrew's SIDARTHE setup used.

Matches forward_pass.py's / simulate_system.py's `dynamics` callable
contract exactly:
    dynamics(dt, xa, u, constants, w, noise_index, augment_states,
             w_from_filter, lam, u_lims, u_lim_method) -> (nxa, K) array
noise_index=False -> deterministic-plus-noise next state.
noise_index=i (1-indexed) -> the i-th column of the noise-diffusion
matrix F_a, broadcast across all K columns (F_a doesn't depend on x/u
here) -- this is what forward_pass.py differentiates to build Cx/Cu.

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - `time_step_method` is hardcoded to 1 (Euler) in the MATLAB source;
    the RK4 branch (time_step_method=2) is dead code that never
    executes (same "confirmed-dead branch, not ported" treatment as
    SPKF_function.m's Mode==1 and iLQG_function.m's full_DDP, elsewhere
    in this port).
  - `lam` (MATLAB's `lambda`) is a genuinely unused parameter in the
    MATLAB source's live code path -- its only appearance is inside a
    commented-out line (`%Fa(:,:,i) = blkdiag(F + lambda^0.33*eye(nx), Fp)`).
    Kept as an unused parameter for signature fidelity with the shared
    `dynamics` contract, same convention as Measurement.m's unused `u`/`c`.
  - The MATLAB's `if any(xkplus1<0)` branch is already dead in the
    source itself (its own warning() call is commented out) and the
    `~isreal(xkplus1)` check can't trigger here (no sqrt/log of a
    state-dependent expression anywhere in this Euler step, unlike
    Andrew's SIDARTHE plant) -- both dropped, same precedent as
    simulate_system.py already dropping this exact isreal check.
"""

import numpy as np

from applications.sci_rehab.continuous_state_dynamics import continuous_state_dynamics


def discrete_state_dynamics(dt, xa, u, constants, w, noise_index,
                               augment_states, w_from_filter, lam, u_lims,
                               u_lim_method):
    Nplus1 = xa.shape[1]

    if w.shape[1] == 1 and Nplus1 > 1:
        w = np.tile(w, (1, Nplus1))
    if u.shape[1] == 1 and Nplus1 > 1:
        u = np.tile(u, (1, Nplus1))

    fa, nx, n_p, F, Fp = continuous_state_dynamics(
        dt, xa, u, constants, augment_states, u_lims, u_lim_method)

    F_a = _block_diag(F, Fp)  # (nxa, nxa), constant across columns

    if noise_index:  # 1-indexed request for the noise-diffusion column
        col = F_a[:, noise_index - 1]
        return np.tile(col[:, None], (1, Nplus1))

    if not w_from_filter:
        noise = (F_a @ w) * np.sqrt(dt)
    else:
        noise = w

    return xa + fa * dt + noise


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
