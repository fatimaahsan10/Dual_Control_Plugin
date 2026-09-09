"""
Adapter from the wizard's generic model config to core/ddp_solver/ilqg.py's
calling convention -- the "iLQR" control method (see schema.CONTROL_METHODS).

NOTE: build_constraint_fn() below (and everything it depends on) is used
by BOTH control methods, not just "ilqr" -- see its own docstring's
"REUSED BY BOTH CONTROL METHODS" section. step_fn/derivs_fn/compile_ilqr/
solve_ilqr remain "ilqr"-only, since the "ilqg" path has its own separate
solver entry point (extensions/dual_control/main_outer_control_loop()),
called directly by wizard/solver_runner.py, not through this module.

This is the counterpart, for the iLQR path, to generic_plant.py+
main_outer_control_loop() for the iLQG path: it is pure GLUE CODE, built
once, generically, from whatever states/actions/constants/dynamics/cost a
given ModelConfig happens to define -- nothing here knows about a robot
arm, a joint, a torque, or any other domain vocabulary. Two existing,
UNMODIFIED pieces of machinery are composed:

  1. wizard/generic_plant.py's compile_plant() -- reused exactly as the
     iLQG path uses it, to get dynamics/cost closures satisfying
     extensions/dual_control/protocols.py's DynamicsFn/CostFn contracts.
     step_fn/derivs_fn below are built by calling THOSE closures with
     augment_states=False (this control method has no unknown parameters
     to estimate -- see schema.ModelConfig.control_method's docstring),
     never by re-deriving dynamics/cost from the equations a second time.

  2. extensions/constraints/ (Dastan & Sensinger 2024, "Zahid's" work) --
     dynamic_control_bounds.build_time_varying_lims/solve_box_from_constraint
     (Variant A, h(x,u)>=0) and relative_degree_reduction.
     state_constraint_to_control_constraint (Variant B, h(x)>=0, reduced
     via one Lie derivative). Called with their existing, documented
     signatures; no constraint algorithm is modified or reimplemented
     here. The only new code is turning a wizard ConstraintSpec's
     equation string into the plain h_fn(x[,u]) callables those functions
     already expect -- the same equation_parser.py machinery Dynamics/
     Measurement/Cost already use, not a new parser.

ANALYTIC JACOBIANS (Plan B refinement over the bare minimum): since a
constraint's expression is parsed into a real sympy expression tree
(exactly like every other wizard equation), differentiating it exactly via
sympy.diff() is cheap and removes one layer of finite-difference noise
relative to always falling back to dynamic_control_bounds.py's/
relative_degree_reduction.py's own numeric-Jacobian fallback (dhdu_fn=None/
dhdx_fn=None). This is supplied through those functions' OWN existing
optional dhdu_fn/dhdx_fn arguments -- it does not change what either
function computes, only how precisely one of their inputs is obtained.
Where an exact analytic Jacobian isn't cheaply available (the FINAL,
h(x,u)-level Jacobian of a "state_only" constraint's reduced h_tilde,
which would require differentiating through the model's own dynamics a
second time), the existing numeric fallback is used deliberately, exactly
as extensions/constraints/relative_degree_reduction.py's own
"fully numeric pipeline" test case already proves works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sympy

from core.ddp_solver.ilqg import ilqg
from extensions.constraints.dynamic_control_bounds import build_time_varying_lims
from extensions.constraints.relative_degree_reduction import (
    state_constraint_to_control_constraint,
)
from extensions.dual_control.finite_difference import finite_difference

from wizard.equation_parser import lambdify_equation, parse_equation
from wizard.generic_plant import CompiledPlant, PlantCompilationError, compile_plant
from wizard.schema import ModelConfig


class ConstraintCompilationError(Exception):
    """Raised by build_constraint_fn() when one or more constraint
    expressions fail to parse/compile. Carries every problem found at
    once, same "report everything, not just the first" convention as
    generic_plant.PlantCompilationError."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# ----------------------------------------------------------------------
# step_fn / derivs_fn -- core.ddp_solver.protocols.StepFn/DerivsFn built
# from the SAME dynamics/cost closures compile_plant() already produces.
# ----------------------------------------------------------------------

def build_step_fn(plant: CompiledPlant, config: ModelConfig) -> Callable:
    """
    StepFn: step_fn(x, u, i) -> (x_next, cost), (n,K)/(m,K) batched.

    Calls plant.dynamics/plant.cost with augment_states=False and
    zero dynamics noise (w=zeros) -- this control method never augments
    (no unknown parameters to estimate; see schema.py) and never injects
    noise (deterministic optimal control). Any parameter defined anyway
    (e.g. a model whose control_method was switched from "ilqg" without
    removing its ParameterSpec rows) is used at its fixed prior_guess
    value via plant.constants.prior -- exactly the same fallback
    generic_plant.py's make_continuous_dynamics_fn already uses whenever
    augment_states=False, unchanged here.
    """
    dt = config.dt
    nx = plant.nx
    u_lims = np.array([[a.min, a.max] for a in config.actions], dtype=float)
    u_lim_method = config.solver.u_lim_method

    def step_fn(x, u, i):
        K = x.shape[1]
        u_safe = np.where(np.isnan(u), 0.0, u)
        w = np.zeros((nx, K))
        x_next = plant.dynamics(
            dt, x, u_safe, plant.constants, w, False, False, False, 0.0,
            u_lims, u_lim_method)
        cost_val = plant.cost(x, u, None, u_lims, u_lim_method, plant.constants)
        return x_next, cost_val

    return step_fn


def build_derivs_fn(plant: CompiledPlant, step_fn: Callable) -> Callable:
    """
    DerivsFn: derivs_fn(x, u_aug) -> (fx, fu, None, None, None, cx, cu,
    cxx, cxu, cuu) -- iLQG mode (full_DDP's fxx/fxu/fuu left None, same
    convention extensions/dual_control/ilqg_function.py already uses).

    Built via finite differences over step_fn, structurally the SAME
    pattern extensions/dual_control/forward_pass.py already uses (reusing
    the same finite_difference.py) to compute its own A/B/q/Q/r/R/P from
    an equivalent dynamics/cost callable pair -- just without that
    module's additional noise-Jacobian bookkeeping (c/Cx/Cu/d/Dx/Du/E/F),
    which core/ddp_solver's deterministic derivs_fn contract has no
    equivalent for.
    """
    nx = plant.nx

    def _dynamics_only(xu):
        x_next, _ = step_fn(xu[:nx, :], xu[nx:, :], 0)
        return x_next

    def _cost_only(xu):
        _, c = step_fn(xu[:nx, :], xu[nx:, :], 0)
        return c.reshape(1, -1)

    def derivs_fn(x, u_aug):
        xu = np.vstack([x, u_aug])

        J = finite_difference(_dynamics_only, xu)
        fx = J[:, :nx, :]
        fu = J[:, nx:, :]

        Jc = finite_difference(_cost_only, xu)[0]
        cx = Jc[:nx, :]
        cu = Jc[nx:, :]

        def _cost_grad(xu_):
            return finite_difference(_cost_only, xu_)[0]

        JJ = finite_difference(_cost_grad, xu)
        JJ = 0.5 * (JJ + JJ.transpose(1, 0, 2))  # symmetrize
        cxx = JJ[:nx, :nx, :]
        cxu = JJ[:nx, nx:, :]
        cuu = JJ[nx:, nx:, :]

        return fx, fu, None, None, None, cx, cu, cxx, cxu, cuu

    return derivs_fn


# ----------------------------------------------------------------------
# Constraints -- generic ConstraintSpec -> core.ddp_solver.protocols.
# ConstraintFn, composed from extensions/constraints/ exactly as-is.
# ----------------------------------------------------------------------

def _parse_constraint_expr(text: str, allowed_names: list[str],
                             const_values: dict[str, float], label: str,
                             errors: list[str]):
    result = parse_equation(text, allowed_names)
    if not result.ok:
        errors.append(f"{label}: {result.error}")
        return None
    try:
        return result.expr.subs(
            {sympy.Symbol(k): v for k, v in const_values.items()})
    except Exception as e:  # noqa: BLE001 -- report, never propagate
        errors.append(f"{label}: could not compile this equation "
                       f"({type(e).__name__}: {e}).")
        return None


def _lambdify_with_gradient(expr, wrt_names: list[str], ordered_names: list[str]):
    """Returns (value_fn, grad_fns) -- grad_fns[i] = d(expr)/d(wrt_names[i]),
    both lambdified against the SAME ordered_names so every resulting
    callable is called with the same positional argument order."""
    value_fn = lambdify_equation(expr, ordered_names)
    grad_fns = [lambdify_equation(sympy.diff(expr, sympy.Symbol(w)), ordered_names)
                for w in wrt_names]
    return value_fn, grad_fns


def _build_state_action_constraint(c, config: ModelConfig,
                                     const_values: dict[str, float],
                                     errors: list[str]):
    """kind == "state_action": h(x, u) -- may reference states, actions,
    constants. Returns (h_fn(x,u)->(1,), dhdu_fn(x,u)->(1,nu)) with an
    EXACT analytic Jacobian (sympy.diff on the user's own expression --
    not a new constraint algorithm, just a precise input to
    dynamic_control_bounds.py's existing optional dhdu_fn argument)."""
    state_names = config.state_names()
    action_names = config.action_names()
    allowed = state_names + action_names + config.constant_names()
    expr = _parse_constraint_expr(c.expression, allowed, const_values,
                                    f"Constraint '{c.name}'", errors)
    if expr is None:
        return None

    ordered = state_names + action_names
    h_num, dhdu_nums = _lambdify_with_gradient(expr, action_names, ordered)

    def h_fn(x, u):
        return np.atleast_1d(np.asarray(h_num(*x, *u), dtype=float))

    def dhdu_fn(x, u):
        row = [float(fn(*x, *u)) for fn in dhdu_nums]
        return np.atleast_2d(np.asarray(row, dtype=float))

    return h_fn, dhdu_fn


def _build_state_only_constraint(c, config: ModelConfig,
                                   const_values: dict[str, float],
                                   errors: list[str]):
    """kind == "state_only": h(x) -- may reference states and constants
    ONLY (the narrower `allowed` list below is what makes referencing an
    action here a plain-language parse error, not a silent ignore).
    Returns (h_fn(x)->(1,), dhdx_fn(x)->(1,nx)), again an exact analytic
    Jacobian of the user's own h(x) -- fed into
    relative_degree_reduction.state_constraint_to_control_constraint's
    existing optional dhdx_fn argument for its internal Lie-derivative
    step; that function's OWN h_tilde = h_dot + alpha*h formula (eq 18-19)
    is untouched."""
    state_names = config.state_names()
    allowed = state_names + config.constant_names()
    expr = _parse_constraint_expr(c.expression, allowed, const_values,
                                    f"Constraint '{c.name}'", errors)
    if expr is None:
        return None

    h_num, dhdx_nums = _lambdify_with_gradient(expr, state_names, state_names)

    def h_fn(x):
        return np.atleast_1d(np.asarray(h_num(*x), dtype=float))

    def dhdx_fn(x):
        row = [float(fn(*x)) for fn in dhdx_nums]
        return np.atleast_2d(np.asarray(row, dtype=float))

    return h_fn, dhdx_fn


def _build_plain_continuous_dynamics(plant: CompiledPlant, config: ModelConfig) -> Callable:
    """dynamics_fn(x, u) -> xdot, (n,)/(m,) single-point, CONTINUOUS-time
    -- the exact interface relative_degree_reduction.lie_derivative_h
    requires (see that module's own docstring: "NOT a discrete one-step
    map"). Built by calling plant.continuous_dynamics (already generic,
    produced by compile_plant() for ANY model) at a single point,
    augment_states=False -- same convention as build_step_fn."""
    nx = plant.nx
    dt = config.dt
    u_lim_method = config.solver.u_lim_method

    def dynamics_fn(x, u):
        xa = np.asarray(x, dtype=float).reshape(-1, 1)
        uu = np.asarray(u, dtype=float).reshape(-1, 1)
        xdot, *_ = plant.continuous_dynamics(
            dt, xa, uu, plant.constants, False, None, u_lim_method)
        return xdot[:nx, 0]

    return dynamics_fn


def build_constraint_fn(plant: CompiledPlant, config: ModelConfig) -> Optional[Callable]:
    """
    Returns None if there are no enabled constraints (the common case --
    every existing wizard model behaves exactly as before). Otherwise
    returns a ConstraintFn (core/ddp_solver/protocols.py) built by:

      1. compiling each enabled ConstraintSpec into a plain h_fn (and,
         for "state_only" rows, reducing it via
         relative_degree_reduction.state_constraint_to_control_constraint
         first);
      2. stacking every row's h_fn into ONE combined h_fn(x,u)->(p,) --
         the same "vector h with independent rows" pattern already
         exercised by extensions/constraints/'s own test suite
         (test_dynamic_control_bounds.py::
         test_vector_h_two_independent_scalar_rows_pin_different_bounds);
      3. handing that combined h_fn to
         dynamic_control_bounds.build_time_varying_lims, with each
         action's existing min/max as the `lower`/`upper` hard-limit
         arguments that function already accepts -- action bounds and
         state-dependent constraints compose through that one existing
         parameter, not a new merging step.

    An exact combined analytic dhdu_fn is supplied ONLY when every
    enabled constraint is "state_action" (no reduction involved, so the
    combined Jacobian is just each row's own exact d h_row/du stacked).
    As soon as any "state_only" row is present, the combined dhdu_fn is
    left None -- dynamic_control_bounds.py's own numeric fallback handles
    the mix (this is the SAME code path
    test_relative_degree_reduction.py::
    test_fully_numeric_pipeline_both_jacobians_from_finite_differences
    already exercises and verifies).

    NOTE on residual gap for a STATE-dependent bound: core.ddp_solver.
    ilqg.py recomputes constraint_fn once per outer iteration from the
    CURRENT nominal trajectory (gated by `flg_change`, per its own
    docstring), then that iteration's backward/forward pass optimizes
    against that fixed box. If the constraint's box depends on a state
    (e.g. h(x,u) referencing a velocity, not just u), the FINAL accepted
    trajectory's own state can differ very slightly from the state the
    box was last linearized around -- observed empirically as a ~1e-3
    magnitude residual gap, not a bug (see wizard/
    test_robot_arm_example.py::test_torque_derating_constraint_actually_
    binds_early's tolerance and comment).

    REUSED BY BOTH CONTROL METHODS: this function is control-method-
    agnostic (its output is just callable(x_traj, u_traj) -> (N,m,2)),
    so wizard/solver_runner.py's "ilqg" branch calls it too, feeding the
    result into extensions/dual_control/main_outer_control_loop()'s own
    (now equally optional) constraint_fn parameter -- see that module's
    and ilqg_function.py's docstrings. ONE real difference between the
    two paths for a "state_only" row: `dyn_fn` above (the continuous
    dynamics used for that row's Lie-derivative reduction) is built via
    `_build_plain_continuous_dynamics`, which always evaluates
    plant.continuous_dynamics with augment_states=False, i.e. at each
    unknown parameter's FIXED prior_guess/true_value (whichever
    compile_plant() put in plant.constants.prior) -- never at the "ilqg"
    path's evolving online parameter ESTIMATE. A "state_only" constraint
    on a model with n_p > 0 is therefore reduced using a slightly-wrong
    (prior, not current-estimate) dynamics model throughout the whole
    dual-control run; a "state_action" constraint is entirely unaffected
    (no dynamics/Lie-derivative involved at all). Not fixed here --
    correctly re-building the reduction from the current p_hat every
    session would require passing p_hat into build_constraint_fn() from
    inside the outer loop itself, a bigger change than this hook's scope;
    flagged rather than silently accepted, same convention as every
    other documented limitation in this file.
    """
    enabled = [c for c in config.constraints if c.enabled]
    if not enabled:
        return None

    const_values = {c.name: c.value for c in config.constants}
    errors: list[str] = []
    h_fns: list[Callable] = []
    analytic_dhdu_rows: list[Optional[Callable]] = []
    any_state_only = False
    dyn_fn: Optional[Callable] = None

    for c in enabled:
        if c.kind == "state_action":
            result = _build_state_action_constraint(c, config, const_values, errors)
            if result is None:
                continue
            h_fn, dhdu_fn = result
            h_fns.append(h_fn)
            analytic_dhdu_rows.append(dhdu_fn)
        else:
            any_state_only = True
            result = _build_state_only_constraint(c, config, const_values, errors)
            if result is None:
                continue
            h_fn_state, dhdx_fn = result
            if dyn_fn is None:
                dyn_fn = _build_plain_continuous_dynamics(plant, config)
            h_tilde_fn = state_constraint_to_control_constraint(
                h_fn_state, dyn_fn, c.alpha, dhdx_fn=dhdx_fn)
            h_fns.append(h_tilde_fn)
            analytic_dhdu_rows.append(None)

    if errors:
        raise ConstraintCompilationError(errors)

    def combined_h_fn(x, u):
        return np.concatenate(
            [np.atleast_1d(np.asarray(f(x, u), dtype=float)) for f in h_fns])

    combined_dhdu_fn = None
    if not any_state_only:
        def combined_dhdu_fn(x, u):
            return np.vstack(
                [np.atleast_2d(np.asarray(f(x, u), dtype=float))
                 for f in analytic_dhdu_rows])

    lower = np.array([a.min for a in config.actions], dtype=float)
    upper = np.array([a.max for a in config.actions], dtype=float)

    def constraint_fn(x_traj, u_traj):
        return build_time_varying_lims(
            combined_h_fn, x_traj, u_traj, dhdu_fn=combined_dhdu_fn,
            lower=lower, upper=upper)

    return constraint_fn


# ----------------------------------------------------------------------
# Full assembly -- the counterpart to generic_plant.compile_plant() +
# main_outer_control_loop() for the iLQG path. Split into
# compile_ilqr()/solve_ilqr() (rather than one function) so
# wizard/solver_runner.py can wrap only the actual solve in its
# timeout/exception-translation machinery, exactly like it already does
# for compile_plant()/main_outer_control_loop().
# ----------------------------------------------------------------------

@dataclass
class IlqrInputs:
    step_fn: Callable
    derivs_fn: Callable
    constraint_fn: Optional[Callable]
    x0: np.ndarray
    u0: np.ndarray
    lims: np.ndarray
    reg_type: int
    max_iter: int


def compile_ilqr(config: ModelConfig) -> IlqrInputs:
    """Raises PlantCompilationError (bad dynamics/cost/measurement
    equations) or ConstraintCompilationError (bad constraint
    expressions) -- never a bare sympy/numpy exception, same convention
    as generic_plant.compile_plant()."""
    plant = compile_plant(config)
    constraint_fn = build_constraint_fn(plant, config)
    step_fn = build_step_fn(plant, config)
    derivs_fn = build_derivs_fn(plant, step_fn)

    x0 = np.array([s.initial_value for s in config.states], dtype=float)
    N = config.n_sessions - 1
    u0 = np.zeros((config.nu, N))
    lims = np.array([[a.min, a.max] for a in config.actions], dtype=float)

    return IlqrInputs(
        step_fn=step_fn, derivs_fn=derivs_fn, constraint_fn=constraint_fn,
        x0=x0, u0=u0, lims=lims, reg_type=config.solver.reg_type,
        max_iter=config.solver.first_run_max_du_iterations)


def solve_ilqr(inputs: IlqrInputs) -> dict:
    """Calls the REAL, UNMODIFIED core.ddp_solver.ilqg.ilqg() and shapes
    its return into the SAME result-dict keys
    extensions/dual_control/main_outer_control_loop() produces
    (x_true/u/cost_true/total_true_cost), so wizard/plotting.py's
    plot_states/plot_actions/plot_cost/results_table work UNCHANGED for
    either control method -- plus two iLQR-only keys (trace/stop_reason)
    for the convergence display iLQG has no equivalent of."""
    x, u, L, Vx, Vxx, cost, trace, stop_reason = ilqg(
        inputs.step_fn, inputs.derivs_fn, inputs.x0, inputs.u0,
        lims=inputs.lims, constraint_fn=inputs.constraint_fn,
        reg_type=inputs.reg_type, max_iter=inputs.max_iter, verbose=0)

    return dict(
        x_true=x, u=u, cost_true=cost, total_true_cost=float(cost.sum()),
        trace=trace, stop_reason=stop_reason)
