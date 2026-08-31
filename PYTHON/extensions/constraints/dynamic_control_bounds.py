"""
Dynamic (state-informed) control bounds for constrained DDP.

Implements Variant A of Dastan & Sensinger (2024), "Leveraging Control
Inputs to Enforce Constraints in Differential Dynamic Programming for
Nonlinear Optimization" (IEEE CDC 2024, DOI 10.1109/CDC56724.2024.10886170)
-- eqs (10)-(13): instead of one fixed box u in [u_min, u_max] for the
whole trajectory (todorov/back_pass.py's `lims`), recompute a
STATE-DEPENDENT box [u_min(x_i), u_max(x_i)] at every timestep i of the
current nominal trajectory, from an inequality constraint h(x, u) >= 0.

No source code for this paper exists (checked -- see project notes): this
is an original implementation derived directly from the paper's
equations, not a port. Verified against the paper's own worked pendulum
example (Sec IV-A, eq 20-23) via a hand-derived closed form -- see
test_dynamic_control_bounds.py.

RESTRICTION (inherited from the paper, not introduced here): this only
handles constraints where each scalar row of h has relative degree one
with respect to EXACTLY ONE control channel -- true of every one of the
paper's own worked examples (m=1 for the pendulum; the 2D car's
obstacle/steering constraints each pin one channel). A row whose Jacobian
has more than one nonzero entry describes a half-space in control space,
not a box edge, so `solve_box_from_constraint` raises rather than
silently dropping information or producing a wrong bound.
"""

import warnings

import numpy as np

from extensions.dual_control.finite_difference import finite_difference

_JAC_ZERO_TOL = 1e-10


def solve_box_from_constraint(h_fn, x_i, u_i, dhdu_fn=None,
                                lower=-np.inf, upper=np.inf, fd_step=2 ** -17):
    """
    Paper eq (10)-(13): given h(x_i, u) >= 0 (p constraint rows) and the
    trajectory point x_i held fixed, first-order-approximate each h_j in u
    around u_i and solve for the tightest box [lo, hi] on u consistent
    with every h_j(x_i, u) >= 0, intersected with pre-existing hard
    limits `lower`/`upper`.

    Parameters
    ----------
    h_fn    : callable(x, u) -> (p,) ndarray.  h(x, u) >= 0 required.
    x_i     : (n,)  state to evaluate/hold fixed -- the "state-informed"
              part of the bound; a parameter here, not a QP variable.
    u_i     : (m,)  control to linearize the Jacobian around (typically
              the current nominal control at this timestep).
    dhdu_fn : callable(x, u) -> (p, m) ndarray, optional analytic
              Jacobian dh/du. If None, computed via forward differences
              using andrew/finite_difference.py, holding x_i fixed and
              differentiating only w.r.t. u.
    lower, upper : scalar or (m,) ndarray, pre-existing hard control
              limits to intersect with the constraint-derived bounds.
              Not discussed by the paper's eq (12) in isolation, but
              nothing there precludes an independently-imposed hard
              actuator limit either -- kept as an explicit, flagged
              generalization rather than assumed silently.
    fd_step : float, forward-difference step passed to finite_difference
              when dhdu_fn is None.

    Returns
    -------
    lo, hi : (m,) ndarrays. Channels no constraint row pins keep whatever
              `lower`/`upper` were passed in (default +-inf).

    Raises
    ------
    ValueError
        if any constraint row's Jacobian has more than one entry whose
        magnitude exceeds `_JAC_ZERO_TOL` -- not expressible as a box
        edge (see module docstring).
    """
    x_i = np.asarray(x_i, dtype=float)
    u_i = np.asarray(u_i, dtype=float)
    m = u_i.shape[0]

    h_val = np.atleast_1d(np.asarray(h_fn(x_i, u_i), dtype=float))
    p = h_val.shape[0]

    if dhdu_fn is not None:
        J = np.atleast_2d(np.asarray(dhdu_fn(x_i, u_i), dtype=float))
    else:
        def h_of_u(U):  # (m, K) -> (p, K)
            K = U.shape[1]
            out = np.empty((p, K))
            for k in range(K):
                out[:, k] = h_fn(x_i, U[:, k])
            return out

        J = finite_difference(h_of_u, u_i.reshape(m, 1), h=fd_step)[:, :, 0]  # (p, m)

    lo = np.broadcast_to(np.asarray(lower, dtype=float), (m,)).copy()
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (m,)).copy()

    for j in range(p):
        nz = np.flatnonzero(np.abs(J[j]) > _JAC_ZERO_TOL)
        if nz.size == 0:
            warnings.warn(
                f"constraint row {j} has zero Jacobian w.r.t. every "
                "control channel at this (x_i, u_i) -- relative degree "
                "is not one here, so this row contributes no bound "
                "(silently unenforced at this point).", RuntimeWarning)
            continue
        if nz.size > 1:
            raise ValueError(
                f"constraint row {j} depends on {nz.size} control "
                f"channels (indices {nz.tolist()}) -- not expressible as "
                "an axis-aligned box bound; the paper's box formulation "
                "requires relative degree one w.r.t. exactly one channel."
            )
        k = int(nz[0])
        u_star = u_i[k] - h_val[j] / J[j, k]
        if J[j, k] < 0:
            hi[k] = min(hi[k], u_star)
        else:
            lo[k] = max(lo[k], u_star)

    return lo, hi


def build_time_varying_lims(h_fn, x_traj, u_traj, dhdu_fn=None,
                              lower=-np.inf, upper=np.inf, fd_step=2 ** -17):
    """
    Paper Fig. 1: loop solve_box_from_constraint over an entire nominal
    trajectory to build the per-timestep bounds array that
    todorov/back_pass.py's and todorov/forward_pass.py's planned
    time-varying `lims` extension expects.

    Parameters
    ----------
    x_traj : (n, N)   nominal state at each of the N control steps --
              i.e. x[:, :N] in ilqg.py's convention (the terminal state
              x[:, N] has no associated control and is excluded).
    u_traj : (m, N)   nominal control at each control step.

    Returns
    -------
    lims : (N, m, 2) ndarray. lims[i, :, 0] = lower bound at control step
              i, lims[i, :, 1] = upper bound -- matches back_pass.py's
              existing `lims[:, 0]` / `lims[:, 1]` indexing, just with a
              leading per-timestep axis.
    """
    n, N = x_traj.shape
    m = u_traj.shape[0]
    lims = np.empty((N, m, 2))
    for i in range(N):
        lo, hi = solve_box_from_constraint(
            h_fn, x_traj[:, i], u_traj[:, i], dhdu_fn, lower, upper, fd_step)
        lims[i, :, 0] = lo
        lims[i, :, 1] = hi
    return lims
