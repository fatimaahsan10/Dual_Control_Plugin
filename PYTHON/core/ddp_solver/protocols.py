"""
Formal plug-in contract for the core/ddp_solver layer (Todorov's
control-limited DDP/iLQG -- ilqg.py/back_pass.py/forward_pass.py).

DOCUMENTATION ONLY at this stage: these typing.Protocol classes describe,
in one place, the callable contracts ilqg.py/back_pass.py/forward_pass.py
already rely on today (transcribed from their own docstrings and from
reading how they actually call step_fn/derivs_fn/constraint_fn -- not
guessed). Nothing in core/ddp_solver imports, isinstance-checks, or
otherwise enforces these yet, and no existing call site is touched by
adding this file. A plant/cost pair (e.g. todorov_toy's car-parking demo,
or a future one) can be checked against this as the single authoritative
source instead of re-deriving the contract from ilqg.py's body.

NAMING GOTCHA, flagged rather than silently normalized: ilqg.py's own
top-level variable `N` means "number of control steps" (`m, N =
u0.shape`), so the trajectory has N+1 points. But back_pass.py's
docstring uses `N` to mean the trajectory-point count (= ilqg.py's N+1)
directly on cx/fx/etc. This file uses ilqg.py's convention (N = control
steps) and writes the derivative arrays' actual column count as `N+1`
explicitly, to avoid perpetuating that clash silently.
"""

from typing import Optional, Protocol

import numpy as np


class StepFn(Protocol):
    """
    Contract for `step_fn`, as consumed by forward_pass.py (and, via it,
    ilqg.py's initial rollout and line search).

    One simulated dynamics+cost step, batched over K candidate
    trajectories at once -- forward_pass.py calls this once per time
    step, evaluating all K backtracking line-search candidates (or all K
    initial-rollout alpha shrinks) in a single call.

    __call__(x, u, i) -> (x_next, cost)

    Parameters
    ----------
    x : (n, K) ndarray
        Current states for K parallel candidates (n = state dim).
    u : (m, K) ndarray
        Controls to apply (m = control dim). At the FINAL step (i == N,
        0-indexed -- one past the last control step) forward_pass.py
        passes u as all-NaN ("final cost only" convention, matching the
        MATLAB DYNCST calling convention). The implementation MUST detect
        this itself (e.g. `np.isnan(u[0])`) and zero it out before using
        it in dynamics/cost -- it is NOT safe to assume u is always
        finite.
    i : int
        0-indexed time step, 0 .. N inclusive. i == N is the terminal
        call described above; for i < N this is an ordinary dynamics
        step.

    Returns
    -------
    x_next : (n, K) ndarray
        Next state. At the terminal call (i == N) forward_pass.py never
        reads this return value (only `cost` is used there) -- but a
        2-tuple must still be returned.
    cost : (K,) ndarray
        Stage cost incurred at this step (terminal/final cost only when
        i == N).
    """

    def __call__(
        self, x: np.ndarray, u: np.ndarray, i: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


class DerivsFn(Protocol):
    """
    Contract for `derivs_fn`, as consumed by ilqg.py (Step 1,
    "differentiate") and fed straight into back_pass.py.

    __call__(x, u) -> (fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu)

    Parameters
    ----------
    x : (n, N+1) ndarray
        Full nominal state trajectory (n = state dim; N = number of
        control steps, i.e. `u0.shape[1]` in ilqg.py).
    u : (m, N+1) ndarray
        Nominal control sequence, right-padded by ilqg.py with one
        all-NaN column (`u_aug`) so its column count matches x's -- the
        last column represents "no control, terminal point" and must be
        handled the same way StepFn's terminal call is (implementation's
        responsibility to zero/ignore it appropriately when computing
        cost derivatives; dynamics derivatives at that column are unused
        by back_pass.py, see below).

    Returns
    -------
    All arrays below are indexed by the same N+1 trajectory points as x
    (both the array with '+1' shown explicitly, matching what
    back_pass.py's `N = cx.shape[1]` actually receives, and NOT ilqg.py's
    own `N` variable which is one smaller):

    fx  : (n, n, N+1) ndarray -- dDynamics/dx.   Last slice unused.
    fu  : (n, m, N+1) ndarray -- dDynamics/du.   Last slice unused.
    fxx : (n, n, n, N+1) ndarray, or None -- d2Dynamics/dx2 (full DDP
          only; None selects iLQG/Gauss-Newton mode -- see back_pass.py).
    fxu : (n, n, m, N+1) ndarray, or None -- d2Dynamics/dxdu.
    fuu : (n, m, m, N+1) ndarray, or None -- d2Dynamics/du2.
    cx  : (n, N+1) ndarray -- dCost/dx (all N+1 points, incl. terminal).
    cu  : (m, N+1) ndarray -- dCost/du. Last column unused.
    cxx : (n, n, N+1) ndarray -- d2Cost/dx2.
    cxu : (n, m, N+1) ndarray -- d2Cost/dxdu.
    cuu : (m, m, N+1) ndarray -- d2Cost/du2.

    fxx/fxu/fuu must be either ALL None (iLQG) or ALL provided (full
    DDP) -- back_pass.py branches on `fxu is not None` / `fuu is not
    None` / `fxx is not None` independently, but ilqg.py's own docstring
    describes this as an iLQG-vs-full-DDP mode switch, not a
    per-array choice.
    """

    def __call__(self, x: np.ndarray, u: np.ndarray) -> tuple[
        Optional[np.ndarray],
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]: ...


class ConstraintFn(Protocol):
    """
    Contract for the optional `constraint_fn` (Dastan & Sensinger 2024
    extension -- see zahid/dynamic_control_bounds.py's
    build_time_varying_lims and zahid/relative_degree_reduction.py's
    state_constraint_to_control_constraint, which are the two supplied
    ways to build one of these).

    __call__(x_traj, u_traj) -> lims

    Parameters
    ----------
    x_traj : (n, N) ndarray
        Current nominal state trajectory, EXCLUDING the terminal point
        (ilqg.py passes `x[:, :N]`, i.e. one column per control step,
        not N+1).
    u_traj : (m, N) ndarray
        Current nominal control sequence (ilqg.py passes `u` directly,
        no NaN padding here).

    Returns
    -------
    lims : (N, m, 2) ndarray
        Per-timestep control box, `lims[i]` = (lower, upper) applying at
        control step i. Recomputed once per ACCEPTED outer iteration
        (gated by ilqg.py's `flg_change`, the same flag that gates
        re-differentiating the dynamics) -- not once per line-search
        candidate. No "unconstrained" sentinel value is recognized for
        this per-timestep shape (unlike the static `(m, 2)` `lims` case);
        to disable constraints, pass `constraint_fn=None` to ilqg()
        instead of trying to encode "unconstrained" through this
        callable's return value.
    """

    def __call__(
        self, x_traj: np.ndarray, u_traj: np.ndarray
    ) -> np.ndarray: ...
