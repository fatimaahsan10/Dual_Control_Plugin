"""
Forward pass (rollout) for iLQG/DDP.

Ported from Todorov's CODES/TODOROV'S CODE/iLQG.m, the local function
`forward_pass` (nested inside iLQG.m).

Simulates the system under a control law

    u_new(i) = u(i) + alpha * du(i) + L(i) @ (x_new(i) - x(i))

for a whole batch of backtracking step sizes `alphas` at once (K = len(alphas)
candidate trajectories simulated in parallel) -- this is what lets the outer
loop's line search try several step sizes with one rollout instead of one
sequential rollout per step size.
"""

import numpy as np


def forward_pass(x0, u, L, x_nom, du, alphas, step_fn, lims=None, diff_fn=None):
    """
    Parameters
    ----------
    x0      : (n,)        initial state
    u       : (m, N)      nominal control sequence
    L       : (m, n, N) or None    feedback gains (None -> no feedback term)
    x_nom   : (n, N) or None       nominal state trajectory the feedback
                                    term measures deviation from (required
                                    if L is given)
    du      : (m, N) or None       feedforward correction k (None -> 0)
    alphas  : (K,)         backtracking step sizes to try, in parallel
    step_fn : callable(x, u, i) -> (x_next, cost)
                x, u are (dim, K) batches, i is the 0-indexed time step;
                x_next is (n, K), cost is (K,).
                At the final step (i == N) u is passed as all-NaN to signal
                "final cost only" -- step_fn must zero it out itself before
                using it in the dynamics/cost, matching the MATLAB DYNCST
                convention.
    lims    : control bounds, or None. Either (m, 2) -- one fixed box
              for the whole horizon -- or (N, m, 2) -- a per-timestep
              box, lims[i] applying at control step i (Dastan & Sensinger
              2024 extension; see zahid/dynamic_control_bounds.py's
              build_time_varying_lims(), which produces exactly this
              shape). No "disable" sentinel for the per-timestep case --
              pass lims=None instead.
    diff_fn : callable(a, b) -> a - b, optional override for state
              subtraction (e.g. to wrap angles); defaults to plain a - b

    Returns
    -------
    xnew : (n, N+1, K)
    unew : (m, N, K)
    cnew : (N+1, K)
    """
    n = x0.shape[0]
    m, N = u.shape
    K = len(alphas)

    xnew = np.zeros((n, N + 1, K))
    unew = np.zeros((m, N, K))
    cnew = np.zeros((N + 1, K))

    xnew[:, 0, :] = x0[:, None]

    for i in range(N):
        ui = np.repeat(u[:, i:i + 1], K, axis=1)

        if du is not None:
            ui = ui + np.outer(du[:, i], alphas)

        if L is not None:
            dx = xnew[:, i, :] - x_nom[:, i:i + 1]
            if diff_fn is not None:
                dx = diff_fn(xnew[:, i, :], np.repeat(x_nom[:, i:i + 1], K, axis=1))
            ui = ui + L[:, :, i] @ dx

        if lims is not None:
            if lims.ndim == 2:
                lo, hi = lims[:, 0:1], lims[:, 1:2]
            else:
                lo, hi = lims[i, :, 0:1], lims[i, :, 1:2]
            ui = np.clip(ui, lo, hi)

        unew[:, i, :] = ui
        xnew[:, i + 1, :], cnew[i, :] = step_fn(xnew[:, i, :], ui, i)

    _, cnew[N, :] = step_fn(xnew[:, N, :], np.full((m, K), np.nan), N)

    return xnew, unew, cnew
