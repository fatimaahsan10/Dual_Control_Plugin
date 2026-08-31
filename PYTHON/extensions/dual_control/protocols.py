"""
Formal plug-in contract for the extensions/dual_control layer (Andrew's
dual/adaptive iLQG extension -- ilqg_function.py/forward_pass.py/
todorov_estimator.py/simulate_system.py/main_outer_control_loop.py).

DOCUMENTATION ONLY at this stage: these typing.Protocol classes describe,
in one place, the plant callable contracts this layer already relies on
today -- transcribed by reading every call site (forward_pass.py,
simulate_system.py, todorov_estimator.py, main_outer_control_loop.py's
dynamics_f/measurement_f closures) and cross-checked against the one
existing concrete plant (applications/sci_rehab/'s
continuous_state_dynamics.py + discrete_state_dynamics.py + measurement.py
+ l_cost.py) to confirm every call site agrees on argument order. Nothing
in extensions/dual_control imports, isinstance-checks, or otherwise
enforces these yet, and no existing call site is touched by adding this
file.

WHAT COUNTS AS THE PLUG-IN SURFACE: a new plant author must supply
DynamicsFn, MeasurementFn, CostFn, and ContinuousDynamicsFn (the four
Protocols below). `simulate_system` (extensions/dual_control/
simulate_system.py) is NOT part of this surface -- it is solver-provided
infrastructure, always reused as-is per the project's existing "reuse as
is" convention (see CLAUDE.md's Katie conversion plan), parameterized
BY the four plant callables rather than being one itself. It is
documented at the bottom of this file for reference only, not as a
Protocol a plant author implements.

SHARED CONVENTIONS across all four Protocols (stated once here rather
than repeated on each):
  - `dt` is a plain float, the integration timestep.
  - `constants` is plant-defined -- Any. In practice this is EITHER a
    custom object (e.g. SindyModelConstants, exposing whatever fields the
    plant's own dynamics/cost need -- see sindy_model_constants.py) OR a
    plain ndarray. Concretely it can even change shape mid-pipeline:
    ilqg_function.py does `constants = np.concatenate([constants,
    p_hat])` when `augment_states=False` (folding the parameter estimate
    into `constants` instead of the state vector). A plant's dynamics/
    measurement/cost implementations are entirely responsible for
    interpreting whatever `constants` holds; this layer never inspects
    it.
  - `noise_index` is `False`, literal `0`, or a 1-indexed `int i`. Call
    sites are NOT consistent about `False` vs. `0` (forward_pass.py/
    simulate_system.py pass `False`; main_outer_control_loop.py's
    dynamics_f/measurement_f closures pass literal `0`) -- both mean
    "no specific noise column requested, give me the deterministic (or
    noise-added) value". A plant implementation MUST branch with a
    truthy check (`if noise_index:`), matching discrete_state_dynamics.py
    -- NOT `if noise_index is False`, which would silently mishandle the
    `0` call sites.
  - `augment_states` is a plain bool: whether `xa`/the returned array
    includes the augmented (estimated-parameter) rows on top of the
    physical state.
  - `u_lims` is `(nu, 2)` or `None`; `u_lim_method` is `1` (bounds
    enforced externally via box-QP -- see back_pass.py's `lims`
    machinery) or `2` (bounds enforced internally, by tanh-squashing u
    inside the plant's own dynamics/cost -- see
    continuous_state_dynamics.py's/l_cost.py's `u_lim_method == 2`
    branches). A plant need not implement method 2 if the pipeline it's
    used in only ever passes method 1, but the parameter must still be
    accepted for signature compatibility.
"""

from typing import Any, Optional, Protocol, Union

import numpy as np


class DynamicsFn(Protocol):
    """
    Contract for `dynamics`, as consumed by forward_pass.py (Jacobian
    construction via finite differencing), simulate_system.py (rollout),
    and main_outer_control_loop.py's `dynamics_f` closure (fed to
    spkf_function.py for the outer re-estimation step).

    __call__(dt, xa, u, constants, w, noise_index, augment_states,
              w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method)
        -> (nxa, K) ndarray

    Parameters
    ----------
    dt : float
    xa : (nxa, K) ndarray
        Current (possibly augmented) state batch.
    u : (nu, K) ndarray
        Control batch.
    constants : Any
        See module docstring.
    w : (nw, K) ndarray when `w_from_filter` is falsy, but (nxa, K)
        ndarray when `w_from_filter` is truthy -- NOT the same shape in
        both modes. Confirmed by tracing main_outer_control_loop.py's
        `dynamics_f` closure into spkf_function.py: the sigma-point
        "process noise" slice it passes has as many rows as the
        (augmented) state itself (`w_hat = np.zeros_like(xa_hat)` at the
        call site), not `nw` rows. `nw` (the noise-CHANNEL count used by
        forward_pass.py's Jacobian loop, `noise_index = 1..nw`) and
        `nxa` (the state dimension) happen to coincide in the one
        existing plant (discrete_state_dynamics.py's F_a is (nxa, nxa)),
        but nothing in this Protocol requires that -- a plant where they
        differ must still accept a (nxa, K) `w` whenever `w_from_filter`
        is truthy. Semantics otherwise depend on `w_from_filter` (below).
    noise_index : False | int
        `False`/`0` -> return the actual next state (deterministic part
        plus whatever noise `w`/`w_from_filter` specify). A 1-indexed
        `int i` -> return the i-th column of the dynamics noise-diffusion
        matrix (constant w.r.t. K, broadcast across all K columns; e.g.
        discrete_state_dynamics.py's F_a doesn't depend on x/u, so it
        tiles one column). This is what forward_pass.py differentiates
        (per noise channel, i = 1..nw) to build the Cx/Cu Jacobians used
        by todorov_estimator.py's effective-noise-covariance terms.
    augment_states : bool
    w_from_filter : 0 | 1 (int, sometimes passed as Python bool)
        0/False -> `w` is a raw noise sample; the implementation scales
        it internally by its own diffusion matrix and `sqrt(dt)` (Euler-
        Maruyama convention -- see discrete_state_dynamics.py:
        `(F_a @ w) * sqrt(dt)`).
        1/True -> `w` IS ALREADY the final additive noise/perturbation
        term (e.g. an SPKF sigma-point offset) -- add it directly with no
        further scaling.
    dyn_noise_reg_lambda : float
        Regularization term folded into the noise scale (callers pass
        `dyn_noise_reg * lam`, or `0.0` for a true/noiseless step). The
        MATLAB-inherited plant (discrete_state_dynamics.py) accepts this
        positionally but does not use it (dead in the source; see that
        module's docstring) -- kept for signature fidelity, not
        necessarily load-bearing in every plant.
    u_lims : (nu, 2) ndarray or None
    u_lim_method : int
        1 or 2 -- see module docstring.

    Returns
    -------
    (nxa, K) ndarray -- next state (noise_index falsy) or a
    noise-diffusion-matrix column broadcast across K (noise_index = i).
    """

    def __call__(
        self,
        dt: float,
        xa: np.ndarray,
        u: np.ndarray,
        constants: Any,
        w: np.ndarray,
        noise_index: Union[bool, int],
        augment_states: bool,
        w_from_filter: Union[bool, int],
        dyn_noise_reg_lambda: float,
        u_lims: Optional[np.ndarray],
        u_lim_method: int,
    ) -> np.ndarray: ...


class MeasurementFn(Protocol):
    """
    Contract for `measurement`, as consumed by forward_pass.py, the
    initial-y_bar computation in ilqg_function.py, todorov_estimator.py
    (deterministic "measurement" rollout), simulate_system.py, and
    main_outer_control_loop.py's `measurement_f` closure.

    __call__(dt, xa, u, constants, v, noise_index, augment_states,
              v_from_filter) -> (ny, K) ndarray

    Same `noise_index`/`v_from_filter` (the measurement-side name for
    `w_from_filter`) conventions as DynamicsFn, but for the OBSERVATION
    model g(x) [+ noise] rather than the state-transition model. Note
    there is no `dyn_noise_reg_lambda`/`u_lims`/`u_lim_method` here --
    the existing plant (measurement.py) never needs control-limit
    squashing on the observation side, and no call site passes these.

    Parameters
    ----------
    dt : float
    xa : (nxa, K) ndarray
    u : (nu, K) ndarray
        Accepted for signature compatibility with every call site, but
        may be entirely unused by a plant whose observation model
        doesn't depend on the control (measurement.py's own `u` is dead
        for exactly this reason -- flagged in its docstring, not a
        contract violation).
    constants : Any
    v : (nv, K) ndarray when `v_from_filter` is falsy, but (ny, K)
        ndarray when `v_from_filter` is truthy -- same asymmetry as
        DynamicsFn's `w`, confirmed the same way: main_outer_control_loop.py's
        `measurement_f` closure passes an SPKF sigma-point slice sized by
        `v_hat = np.zeros(ny)` at the call site, i.e. `ny` rows, not `nv`.
        `nv` and `ny` coincide in the one existing plant (measurement.py
        observes SCIM/BBS directly, one noise channel per observed
        output), but nothing in this Protocol requires that in general.
    noise_index : False | int
    augment_states : bool
    v_from_filter : 0 | 1

    Returns
    -------
    (ny, K) ndarray -- observation (noise_index falsy) or a
    noise-diffusion-matrix column broadcast across K (noise_index = i).
    """

    def __call__(
        self,
        dt: float,
        xa: np.ndarray,
        u: np.ndarray,
        constants: Any,
        v: np.ndarray,
        noise_index: Union[bool, int],
        augment_states: bool,
        v_from_filter: Union[bool, int],
    ) -> np.ndarray: ...


class CostFn(Protocol):
    """
    Contract for `cost`, as consumed by forward_pass.py (differentiated
    via finite differences, twice, for the cost Hessian) and called
    directly by ilqg_function.py/main_outer_control_loop.py to score
    realized trajectories.

    __call__(xa, u, tracking_trajectory, u_lims, u_lim_method, constants)
        -> (K,) ndarray

    Parameters
    ----------
    xa : (nxa, K) ndarray
    u : (nu, K) ndarray
        Columns may be all-NaN to signal "final/terminal cost only" (same
        convention as core/ddp_solver's StepFn) -- the implementation
        MUST zero these out itself before using them (e.g.
        `l_cost.py`'s `u[:, final] = 0.0` after detecting
        `np.isnan(u[0, :])`); it is not safe to assume u is finite.
    tracking_trajectory : Any (commonly None)
        Reserved for a time-varying reference trajectory. The one
        existing plant (l_cost.py) never uses it (fixed-ceiling cost, not
        trajectory-tracking) -- kept for signature fidelity; a future
        tracking-style cost would read it.
    u_lims : (nu, 2) ndarray or None
    u_lim_method : int
        1 or 2 -- see module docstring. Method 2 implementations must
        apply the SAME tanh-squash to `u` internally that a method-2
        DynamicsFn applies, so cost and dynamics agree on what "the
        control" physically means (see l_cost.py's `u_lim_method == 2`
        branch, which mirrors continuous_state_dynamics.py's).
    constants : Any

    Returns
    -------
    (K,) ndarray -- total cost (state + resource [+ terminal]) per
    column.
    """

    def __call__(
        self,
        xa: np.ndarray,
        u: np.ndarray,
        tracking_trajectory: Any,
        u_lims: Optional[np.ndarray],
        u_lim_method: int,
        constants: Any,
    ) -> np.ndarray: ...


class ContinuousDynamicsFn(Protocol):
    """
    Contract for `continuous_dynamics`, called ONLY by
    main_outer_control_loop.py (not by ilqg_function.py/forward_pass.py,
    which instead call the discrete `dynamics`) -- used once per outer
    MPC iteration, when `augment_states_in_filter` is True, purely to
    obtain the parameter noise-scale matrix `Fp` for constructing `Pw`
    ahead of the SPKF re-estimation step.

    __call__(dt, xa, u, constants, augment_states, u_lims, u_lim_method)
        -> (xdot, nx, n_p, F, Fp)

    Parameters
    ----------
    dt : float
    xa : (nxa, K) ndarray
    u : (nu, K) ndarray
    constants : Any
    augment_states : bool
    u_lims : (nu, 2) ndarray or None
    u_lim_method : int

    Returns
    -------
    xdot : (nx [+ n_p], K) ndarray
        Continuous-time state (and, if augmented, parameter) derivatives
        -- NOT a discrete next-state; this is dx/dt = f(x, u), separate
        from whatever Euler/RK integration DynamicsFn performs on top of
        it (see continuous_state_dynamics.py vs. discrete_state_dynamics.py
        for the existing plant's version of this split -- the latter
        calls the former and Euler-integrates the result; that call
        relationship is a design choice of the existing plant, not part
        of this Protocol's requirement).
    nx : int
        Physical (non-augmented) state dimension. Constant across calls.
    n_p : int
        Number of augmented (estimated) parameters; 0 if
        `augment_states` is False.
    F : (nx, nx) ndarray
        Dynamics noise scale matrix for the physical states.
    Fp : (n_p, n_p) ndarray
        Parameter noise scale matrix; shape `(0, 0)` if `augment_states`
        is False.
    """

    def __call__(
        self,
        dt: float,
        xa: np.ndarray,
        u: np.ndarray,
        constants: Any,
        augment_states: bool,
        u_lims: Optional[np.ndarray],
        u_lim_method: int,
    ) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]: ...


class SimulateSystemFn(Protocol):
    """
    NOT part of the plug-in surface a plant author implements -- see
    module docstring. Documented here only because ilqg_function.py takes
    it as an explicit parameter (`simulate_system_fn`) rather than
    importing extensions/dual_control/simulate_system.py directly, so its
    contract is worth recording alongside the four real plug-in points it
    is parameterized by.

    __call__(dt, xa0, U, constants, sqrtR, sqrtQ, augment_states, lam,
              u_lims, u_lim_method, dynamics, measurement) -> (y, xa)

    Parameters
    ----------
    dt : float
    xa0 : (nxa,) ndarray
        Initial state (single trajectory, not a K-batch).
    U : (nu, M) ndarray
        Control sequence, M >= 1.
    constants : Any
    sqrtR : (nv, nv) ndarray
        Cholesky factor of measurement-noise covariance (pass
        `0.0 * sqrtR` for a noiseless/deterministic rollout).
    sqrtQ : (nw, nw) ndarray
        Cholesky factor of dynamics-noise covariance (same
        zero-for-deterministic convention).
    augment_states : bool
    lam : float
        Forwarded to `dynamics` as its `dyn_noise_reg_lambda`.
    u_lims : (nu, 2) ndarray or None
    u_lim_method : int
    dynamics : DynamicsFn
    measurement : MeasurementFn

    Returns
    -------
    y : (ny, M+1) ndarray
    xa : (nxa, M+1) ndarray
    """

    def __call__(
        self,
        dt: float,
        xa0: np.ndarray,
        U: np.ndarray,
        constants: Any,
        sqrtR: np.ndarray,
        sqrtQ: np.ndarray,
        augment_states: bool,
        lam: float,
        u_lims: Optional[np.ndarray],
        u_lim_method: int,
        dynamics: DynamicsFn,
        measurement: MeasurementFn,
    ) -> tuple[np.ndarray, np.ndarray]: ...
