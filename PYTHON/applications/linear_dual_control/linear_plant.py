"""
Andrew/Mathis PhD thesis Sec. 3.4 linear dual-control example -- the toy
problem that precedes the SIDARTHE testbed in the thesis. Until now,
Andrew's entire Python contribution was extensions/dual_control/ (the
solver core) with no applications/-level plant of his own (see
CLAUDE.md's Repository layout: "Andrew has no applications/ entry ...
the SIDARTHE COVID plant was never ported to Python"). This is that
plant, for the ONE Andrew-thesis figure that doesn't require SIDARTHE.

Continuous dynamics (thesis Sec. 3.4):
    x1_dot = d1*u1 - d2
    x2_dot = d2*u2
d = [d1, d2] is an UNKNOWN constant plant parameter, estimated online by
augmenting it onto the state vector: xa = [x1, x2, d1, d2] (the whole
point of the dual-control demo -- the controller must trade off
exploiting its current d-estimate against exploring to reduce
uncertainty in it).

Cost (thesis Sec. 3.4):
    J(x, u) = (x1 - 5)^2 + 1e-4 * u2^2
Drives x1 to a setpoint of 5; only u2 is resource-penalized (u1 is a
free/cheap control -- it's the channel that actually cancels the -d2
offset and drives x1 to 5, while u2 only feeds back into x2, which
carries no direct cost term of its own other than through this
control-effort penalty).

No control limits are stated in the thesis for this example. Callers
should pass a very wide `u_lims` with `u_lim_method=1` (external box-QP
bound, effectively never binding) rather than `u_lim_method=2`'s tanh
squash, which isn't part of this problem.

Noise conventions match the codebase's one existing production plant
(applications/sci_rehab/discrete_state_dynamics.py /
applications/sci_rehab/measurement.py): a "raw" noise sample
(w_from_filter=0 / v_from_filter=0) is scaled internally by this
plant's own diffusion matrix and sqrt(dt) (Euler-Maruyama); an
"already-final" sample (w_from_filter=1 / v_from_filter=1, i.e. an SPKF
sigma-point offset) is added directly. F_STATE/F_MEAS below are taken
directly from the thesis's stated F=1e-15 (process noise) and G=1e-2
(measurement noise) as diffusion-loading magnitudes (matching this
codebase's existing convention of small constant loadings, e.g.
extensions/dual_control/test_main_outer_control_loop.py's
`col[...] = 0.02`), not as variances to be square-rooted again.
"""

import numpy as np

NX = 2   # x1, x2
NU = 2   # u1, u2
N_P = 2  # d1, d2 (estimated parameters)
NY = 2   # both states observed directly
NV = 2

F_STATE = 1e-15  # thesis: process noise
F_PARAM = 1e-15  # d1, d2 are genuinely constant here; kept tiny rather
                  # than exactly 0 only to keep the SPKF's Cholesky
                  # factorizations well-posed.
G_MEAS = 1e-2    # thesis: measurement noise


def _params(xa, u, constants, augment_states):
    K = xa.shape[1]
    if augment_states:
        return xa[NX:NX + N_P, :]
    return np.asarray(constants[-N_P:]).reshape(N_P, 1) * np.ones((N_P, K))


def dynamics(dt, xa, u, constants, w, noise_index, augment_states,
              w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method):
    K = xa.shape[1]
    if u.shape[1] == 1 and K > 1:
        u = np.tile(u, (1, K))

    nxa = xa.shape[0]
    diag = [F_STATE, F_STATE] + ([F_PARAM] * N_P if augment_states else [])
    F_a = np.diag(diag)

    if not noise_index:
        x = xa[:NX, :]
        d = _params(xa, u, constants, augment_states)
        d1, d2 = d[0:1, :], d[1:2, :]

        x1_next = x[0:1, :] + dt * (d1 * u[0:1, :] - d2)
        x2_next = x[1:2, :] + dt * (d2 * u[1:2, :])
        x_next = np.vstack([x1_next, x2_next])
        xa_next = np.vstack([x_next, xa[NX:, :]]) if augment_states else x_next

        if w_from_filter:
            xa_next = xa_next + w
        else:
            xa_next = xa_next + (F_a @ w) * np.sqrt(dt)
        return xa_next
    else:
        col = np.zeros((nxa, K))
        col[noise_index - 1, :] = F_a[noise_index - 1, noise_index - 1]
        return col


def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                  v_from_filter):
    K = xa.shape[1]
    if noise_index:
        col = np.zeros((NY, K))
        col[noise_index - 1, :] = G_MEAS
        return col

    x = xa[:NX, :]
    if v_from_filter:
        noise = v
    else:
        noise = (G_MEAS * v) * np.sqrt(dt)
    return x + noise


def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
    x1 = xa[0, :]
    u_clean = np.where(np.isnan(u), 0.0, u)
    u2 = u_clean[1, :]
    return (x1 - 5.0) ** 2 + 1e-4 * u2 ** 2


def continuous_dynamics(dt, xa, u, constants, augment_states, u_lims,
                          u_lim_method):
    K = xa.shape[1]
    x = xa[:NX, :]
    n_p = N_P if augment_states else 0
    d = _params(xa, u, constants, augment_states)
    d1, d2 = d[0:1, :], d[1:2, :]

    x1_dot = d1 * u[0:1, :] - d2
    x2_dot = d2 * u[1:2, :]
    xdot = np.vstack([x1_dot, x2_dot])
    if augment_states:
        xdot = np.vstack([xdot, np.zeros((N_P, K))])

    F = F_STATE * np.eye(NX)
    Fp = F_PARAM * np.eye(n_p) if augment_states else np.zeros((0, 0))
    return xdot, NX, n_p, F, Fp
