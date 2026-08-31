"""
Pre-flight validation for a candidate extensions/dual_control plug-in
(dynamics, measurement, cost, continuous_dynamics), checked against the
contracts formally documented in protocols.py.

Purpose: catch a mismatched signature or wrong-shaped/NaN-leaking return
value with ONE clear, specific error BEFORE handing the callables to
ilqg_function()/main_outer_control_loop() -- instead of a wrong answer
propagating silently through forward_pass.py's finite-difference
Jacobians or todorov_estimator.py's covariance recursions, surfacing (if
at all) as an opaque numerical failure with no link back to the actual
mistake.

Entirely opt-in: call validate_dual_control_plugin(...) yourself before
wiring a new plant into main_outer_control_loop(). Nothing in
ilqg_function.py/forward_pass.py/main_outer_control_loop.py calls this
automatically, and none of them are modified by adding this file.

Reuses core/ddp_solver/plugin_validation.py's generic helpers -- same
cross-layer reuse already established by backward_pass.py importing
core/ddp_solver/box_qp.py directly.

`constants` MUST be supplied by the caller as a real constants object
for the plant under test (e.g. a SindyModelConstants instance) -- see
protocols.py's module docstring for why this can't be synthesized
generically; it is plant-defined and this module never inspects it.

Scope: SHAPE / SIGNATURE / NaN-hygiene / documented-convention checks on
synthetic data, not a correctness proof -- this cannot tell you the
plant's physics are right, only that the callables honor their
documented calling convention.
"""

import numpy as np

from core.ddp_solver.plugin_validation import (
    PluginContractError, ValidationIssue, check_arity, check_no_nan,
    check_return_tuple, check_shape, safe_call)


def _default_u_lims(nu: int) -> np.ndarray:
    return np.column_stack([np.zeros(nu), np.ones(nu)])


def validate_dynamics_fn(dynamics, *, nx: int, nu: int, nv: int, nw=None,
                            constants, n_p: int = 2, K: int = 3,
                            u_lims=None, u_lim_method: int = 1,
                            augment_states_modes=(True, False),
                            rng: np.random.Generator = None) -> list:
    """
    Validates against protocols.DynamicsFn.

    `nw` (the dynamics noise-CHANNEL count, i.e. `w`'s row count when
    `w_from_filter` is falsy) is OPTIONAL. Left as None (the default),
    it is recomputed for EACH `augment_states` mode as `nx + n_p if
    augment else nx` -- matching main_outer_control_loop.py's own
    default rule (`nw = nx + n_p if augment_states_in_ilqg else nx`) --
    since for the one existing plant, the noise-diffusion matrix F_a
    literally shrinks from (nx+n_p, nx+n_p) to (nx, nx) when
    `augment_states=False` (Fp becomes (0,0)), so a single fixed `nw`
    tested across both modes would wrongly report a shape mismatch in
    whichever mode it doesn't match. Pass a fixed int here ONLY if your
    plant genuinely uses the same noise-channel count regardless of
    `augment_states` (uncommon -- not true of the existing SCI plant).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    issues: list = []
    fn_label = "dynamics"
    param_names = ["dt", "xa", "u", "constants", "w", "noise_index",
                   "augment_states", "w_from_filter", "dyn_noise_reg_lambda",
                   "u_lims", "u_lim_method"]
    if not check_arity(dynamics, param_names, fn_label, issues):
        return issues
    if u_lims is None:
        u_lims = _default_u_lims(nu)

    for augment in augment_states_modes:
        nxa = nx + n_p if augment else nx
        nw_mode = nw if nw is not None else nxa
        tag = f"augment_states={augment}"
        xa = rng.uniform(0.0, 1.0, size=(nxa, K))
        u = rng.uniform(0.0, 1.0, size=(nu, K))

        check = f"deterministic call ({tag})"
        result = safe_call(dynamics, (0.1, xa, u, constants, np.zeros((nw_mode, K)),
                                        False, augment, 0, 0.0, u_lims, u_lim_method),
                             fn_label, check, issues)
        if result is not None:
            check_shape(result, (nxa, K), "return value", fn_label, check, issues)

        check = f"noise_index False vs 0 consistency ({tag})"
        r_false = safe_call(dynamics, (0.1, xa, u, constants, np.zeros((nw_mode, K)),
                                         False, augment, 0, 0.0, u_lims, u_lim_method),
                              fn_label, check, issues)
        r_zero = safe_call(dynamics, (0.1, xa, u, constants, np.zeros((nw_mode, K)),
                                        0, augment, 0, 0.0, u_lims, u_lim_method),
                             fn_label, check, issues)
        if (isinstance(r_false, np.ndarray) and isinstance(r_zero, np.ndarray)
                and r_false.shape == r_zero.shape
                and not np.allclose(r_false, r_zero, equal_nan=True)):
            issues.append(ValidationIssue(
                fn_label, check,
                "dynamics(..., noise_index=False, ...) and dynamics(..., "
                "noise_index=0, ...) returned different results -- call "
                "sites in this codebase pass BOTH False and literal 0 to "
                "mean 'no noise column requested' (see DynamicsFn Protocol "
                "docstring); implementations must branch with `if "
                "noise_index:` (truthy), not `if noise_index is False`"))

        check = f"w_from_filter 0 vs 1 consistency at w=0 ({tag})"
        r0 = safe_call(dynamics, (0.1, xa, u, constants, np.zeros((nw_mode, K)),
                                    False, augment, 0, 0.0, u_lims, u_lim_method),
                         fn_label, check, issues)
        r1 = safe_call(dynamics, (0.1, xa, u, constants, np.zeros((nxa, K)),
                                    False, augment, 1, 0.0, u_lims, u_lim_method),
                         fn_label, check, issues)
        if (isinstance(r0, np.ndarray) and isinstance(r1, np.ndarray)
                and r0.shape == r1.shape
                and not np.allclose(r0, r1, equal_nan=True)):
            issues.append(ValidationIssue(
                fn_label, check,
                "with w=0 (no actual noise), dynamics() disagreed between "
                "w_from_filter=0 (raw-noise-to-be-scaled convention, w shape "
                "(nw,K)) and w_from_filter=1 (already-final-noise convention, "
                "w shape (nxa,K) -- matching main_outer_control_loop.py's "
                "dynamics_f SPKF closure) -- at zero noise these must "
                "produce the identical deterministic next state regardless "
                "of which convention is used"))

        if nw_mode > 0:
            check = f"noise-diffusion column request ({tag})"
            result = safe_call(dynamics, (0.1, xa, u, constants, np.zeros((nw_mode, K)),
                                            1, augment, 0, 0.0, u_lims, u_lim_method),
                                 fn_label, check, issues)
            if result is not None:
                check_shape(result, (nxa, K), "return value", fn_label, check, issues)

    return issues


def validate_measurement_fn(measurement, *, nx: int, nu: int, nv: int, ny: int,
                               constants, n_p: int = 2, K: int = 3,
                               augment_states_modes=(True, False),
                               rng: np.random.Generator = None) -> list:
    """Validates against protocols.MeasurementFn."""
    rng = rng if rng is not None else np.random.default_rng(0)
    issues: list = []
    fn_label = "measurement"
    param_names = ["dt", "xa", "u", "constants", "v", "noise_index",
                   "augment_states", "v_from_filter"]
    if not check_arity(measurement, param_names, fn_label, issues):
        return issues

    for augment in augment_states_modes:
        nxa = nx + n_p if augment else nx
        tag = f"augment_states={augment}"
        xa = rng.uniform(0.0, 1.0, size=(nxa, K))
        u = rng.uniform(0.0, 1.0, size=(nu, K))

        check = f"deterministic call ({tag})"
        result = safe_call(measurement, (0.1, xa, u, constants, np.zeros((nv, K)),
                                           False, augment, 0), fn_label, check, issues)
        if result is not None:
            check_shape(result, (ny, K), "return value", fn_label, check, issues)

        check = f"noise_index False vs 0 consistency ({tag})"
        r_false = safe_call(measurement, (0.1, xa, u, constants, np.zeros((nv, K)),
                                            False, augment, 0), fn_label, check, issues)
        r_zero = safe_call(measurement, (0.1, xa, u, constants, np.zeros((nv, K)),
                                           0, augment, 0), fn_label, check, issues)
        if (isinstance(r_false, np.ndarray) and isinstance(r_zero, np.ndarray)
                and r_false.shape == r_zero.shape
                and not np.allclose(r_false, r_zero, equal_nan=True)):
            issues.append(ValidationIssue(
                fn_label, check,
                "measurement(..., noise_index=False, ...) and measurement(..., "
                "noise_index=0, ...) returned different results -- both must "
                "mean 'no noise column requested' (see MeasurementFn Protocol "
                "docstring); implementations must branch with `if "
                "noise_index:` (truthy), not `if noise_index is False`"))

        check = f"v_from_filter 0 vs 1 consistency at v=0 ({tag})"
        r0 = safe_call(measurement, (0.1, xa, u, constants, np.zeros((nv, K)),
                                       False, augment, 0), fn_label, check, issues)
        r1 = safe_call(measurement, (0.1, xa, u, constants, np.zeros((ny, K)),
                                       False, augment, 1), fn_label, check, issues)
        if (isinstance(r0, np.ndarray) and isinstance(r1, np.ndarray)
                and r0.shape == r1.shape
                and not np.allclose(r0, r1, equal_nan=True)):
            issues.append(ValidationIssue(
                fn_label, check,
                "with v=0 (no actual noise), measurement() disagreed between "
                "v_from_filter=0 (raw-noise convention, v shape (nv,K)) and "
                "v_from_filter=1 (already-final-noise convention, v shape "
                "(ny,K) -- matching main_outer_control_loop.py's "
                "measurement_f SPKF closure) -- at zero noise these must "
                "agree regardless of which convention is used"))

        if nv > 0:
            check = f"noise-diffusion column request ({tag})"
            result = safe_call(measurement, (0.1, xa, u, constants, np.zeros((nv, K)),
                                               1, augment, 0), fn_label, check, issues)
            if result is not None:
                check_shape(result, (ny, K), "return value", fn_label, check, issues)

    return issues


def validate_cost_fn(cost, *, nx: int, nu: int, constants, n_p: int = 2,
                        K: int = 3, u_lims=None, u_lim_method: int = 1,
                        tracking_trajectory=None,
                        augment_states_modes=(True, False),
                        rng: np.random.Generator = None) -> list:
    """Validates against protocols.CostFn."""
    rng = rng if rng is not None else np.random.default_rng(0)
    issues: list = []
    fn_label = "cost"
    param_names = ["xa", "u", "tracking_trajectory", "u_lims", "u_lim_method",
                   "constants"]
    if not check_arity(cost, param_names, fn_label, issues):
        return issues
    if u_lims is None:
        u_lims = _default_u_lims(nu)

    for augment in augment_states_modes:
        nxa = nx + n_p if augment else nx
        tag = f"augment_states={augment}"
        xa = rng.uniform(0.0, 1.0, size=(nxa, K))
        u = rng.uniform(0.0, 1.0, size=(nu, K))

        check = f"ordinary call ({tag})"
        result = safe_call(cost, (xa, u, tracking_trajectory, u_lims,
                                    u_lim_method, constants), fn_label, check, issues)
        if result is not None and check_shape(result, (K,), "return value",
                                                fn_label, check, issues):
            check_no_nan(result, "return value", fn_label, check, issues)

        check = f"terminal call, u[:, 0]=NaN ({tag})"
        u_terminal = u.copy()
        u_terminal[:, 0] = np.nan
        result = safe_call(cost, (xa, u_terminal, tracking_trajectory, u_lims,
                                    u_lim_method, constants), fn_label, check, issues)
        if result is not None and check_shape(result, (K,), "return value",
                                                fn_label, check, issues):
            if np.isnan(result).any():
                issues.append(ValidationIssue(
                    fn_label, check,
                    "cost contains NaN when a control column is the "
                    "documented all-NaN terminal marker -- cost() must "
                    "detect and zero the NaN control column itself (see "
                    "CostFn Protocol docstring) before using it; this is "
                    "the single most common plug-in bug"))
    return issues


def validate_continuous_dynamics_fn(continuous_dynamics, *, nx: int, nu: int,
                                        constants, n_p: int = 2, K: int = 3,
                                        u_lims=None, u_lim_method: int = 1,
                                        augment_states_modes=(True, False),
                                        rng: np.random.Generator = None) -> list:
    """Validates against protocols.ContinuousDynamicsFn."""
    rng = rng if rng is not None else np.random.default_rng(0)
    issues: list = []
    fn_label = "continuous_dynamics"
    param_names = ["dt", "xa", "u", "constants", "augment_states", "u_lims",
                   "u_lim_method"]
    if not check_arity(continuous_dynamics, param_names, fn_label, issues):
        return issues
    if u_lims is None:
        u_lims = _default_u_lims(nu)

    for augment in augment_states_modes:
        nxa = nx + n_p if augment else nx
        tag = f"augment_states={augment}"
        xa = rng.uniform(0.0, 1.0, size=(nxa, K))
        u = rng.uniform(0.0, 1.0, size=(nu, K))
        expected_n_p = n_p if augment else 0

        check = f"call ({tag})"
        result = safe_call(continuous_dynamics, (0.1, xa, u, constants, augment,
                                                    u_lims, u_lim_method),
                             fn_label, check, issues)
        if result is None:
            continue
        if not check_return_tuple(result, 5, ["xdot", "nx", "n_p", "F", "Fp"],
                                    fn_label, check, issues):
            continue
        xdot, nx_ret, n_p_ret, F, Fp = result

        check_shape(xdot, (nxa, K), "xdot", fn_label, check, issues)
        if nx_ret != nx:
            issues.append(ValidationIssue(
                fn_label, check,
                f"nx: expected {nx} (as declared by the caller), got {nx_ret}"))
        if n_p_ret != expected_n_p:
            issues.append(ValidationIssue(
                fn_label, check,
                f"n_p: expected {expected_n_p} "
                f"({'augmented' if augment else 'not augmented'}), got {n_p_ret}"))
        check_shape(F, (nx, nx), "F", fn_label, check, issues)
        check_shape(Fp, (expected_n_p, expected_n_p), "Fp", fn_label, check, issues)

    return issues


def validate_dual_control_plugin(dynamics, measurement, cost,
                                    continuous_dynamics, *, nx: int, nu: int,
                                    nv: int, ny: int, constants, nw=None,
                                    n_p: int = 2, K: int = 3,
                                    rng: np.random.Generator = None) -> None:
    """
    Runs every applicable check across all four plug-in callables and
    raises ONE PluginContractError listing every issue found, or returns
    silently (no output) if the plug-in is fully contract-compliant.

    `constants` must be a real constants object for the plant under
    test, shared across all four calls -- see this module's docstring
    for why it can't be synthesized generically.

    `nw` is passed through to validate_dynamics_fn unchanged -- leave it
    None (the default) unless your plant's noise-channel count is fixed
    independent of `augment_states`; see validate_dynamics_fn's
    docstring for why that's the uncommon case.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    issues = []
    issues += validate_dynamics_fn(dynamics, nx=nx, nu=nu, nv=nv, nw=nw,
                                      constants=constants, n_p=n_p, K=K, rng=rng)
    issues += validate_measurement_fn(measurement, nx=nx, nu=nu, nv=nv, ny=ny,
                                         constants=constants, n_p=n_p, K=K, rng=rng)
    issues += validate_cost_fn(cost, nx=nx, nu=nu, constants=constants,
                                  n_p=n_p, K=K, rng=rng)
    issues += validate_continuous_dynamics_fn(continuous_dynamics, nx=nx, nu=nu,
                                                 constants=constants, n_p=n_p,
                                                 K=K, rng=rng)
    if issues:
        raise PluginContractError(issues)
