"""
Pre-flight validation for a candidate core/ddp_solver plug-in (step_fn,
derivs_fn, and optionally constraint_fn), checked against the contracts
formally documented in protocols.py.

Purpose: catch a mismatched signature or wrong-shaped/NaN-leaking return
value with ONE clear, specific error BEFORE handing the callable to
ilqg() -- instead of it silently producing a wrong answer, or crashing
several functions later inside back_pass.py's Cholesky solve with a
confusing LinAlgError that doesn't point back to the actual mistake.

Entirely opt-in: call validate_core_plugin(...) yourself before ilqg(...)
in your own demo/test script. Nothing in ilqg.py/back_pass.py/
forward_pass.py calls this automatically, and none of them are modified
by adding this file.

Scope: this is a SHAPE / SIGNATURE / NaN-hygiene check on synthetic
random data (see protocols.py's own docstrings for where each shape
comes from), not a correctness proof -- it cannot tell you your dynamics
or cost are physically right, only that the callable honors its
documented calling convention well enough that ilqg() won't misuse it.
"""

import numpy as np

from core.ddp_solver.plugin_validation import (
    PluginContractError, ValidationIssue, check_arity, check_no_nan,
    check_return_tuple, check_shape, safe_call)


def validate_step_fn(step_fn, *, n: int, m: int, N: int, K: int = 3,
                       rng: np.random.Generator = None) -> list:
    """
    Validates against protocols.StepFn. n/m/N/K follow ilqg.py's own
    convention: n = state dim, m = control dim, N = number of control
    steps (so the trajectory has N+1 points), K = batch size (number of
    parallel line-search candidates forward_pass.py would evaluate).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    issues: list = []
    fn_label = "step_fn"
    if not check_arity(step_fn, ["x", "u", "i"], fn_label, issues):
        return issues

    check = "ordinary step (i=0)"
    x = rng.standard_normal((n, K))
    u = rng.standard_normal((m, K))
    result = safe_call(step_fn, (x, u, 0), fn_label, check, issues)
    if result is not None and check_return_tuple(
            result, 2, ["x_next", "cost"], fn_label, check, issues):
        x_next, cost = result
        if check_shape(x_next, (n, K), "x_next", fn_label, check, issues):
            check_no_nan(x_next, "x_next", fn_label, check, issues)
        if check_shape(cost, (K,), "cost", fn_label, check, issues):
            check_no_nan(cost, "cost", fn_label, check, issues)

    check = "terminal step (i=N, u=NaN)"
    u_terminal = np.full((m, K), np.nan)
    result = safe_call(step_fn, (x, u_terminal, N), fn_label, check, issues)
    if result is not None and check_return_tuple(
            result, 2, ["x_next", "cost"], fn_label, check, issues):
        _, cost = result
        if check_shape(cost, (K,), "cost", fn_label, check, issues) and np.isnan(cost).any():
            issues.append(ValidationIssue(
                fn_label, check,
                "cost contains NaN when called with the documented all-NaN "
                "terminal control -- step_fn must detect and zero the NaN "
                "control itself (see StepFn Protocol docstring) before using "
                "it; this is the single most common plug-in bug"))
    return issues


def validate_derivs_fn(derivs_fn, *, n: int, m: int, N: int,
                          rng: np.random.Generator = None) -> list:
    """Validates against protocols.DerivsFn."""
    rng = rng if rng is not None else np.random.default_rng(0)
    issues: list = []
    fn_label = "derivs_fn"
    if not check_arity(derivs_fn, ["x", "u"], fn_label, issues):
        return issues

    P = N + 1  # trajectory-point count -- see protocols.py's naming-gotcha note
    check = "call"
    x = rng.standard_normal((n, P))
    u = rng.standard_normal((m, P))
    u[:, -1] = np.nan  # documented terminal-column convention

    names = ["fx", "fu", "fxx", "fxu", "fuu", "cx", "cu", "cxx", "cxu", "cuu"]
    result = safe_call(derivs_fn, (x, u), fn_label, check, issues)
    if result is None or not check_return_tuple(result, 10, names, fn_label, check, issues):
        return issues
    fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu = result

    # cx/cxx: back_pass.py reads these directly as the terminal boundary
    # condition (Vx[:,-1]=cx[:,-1], Vxx[:,:,-1]=cxx[:,:,-1]) -- ALL P
    # columns, including the last, must be finite.
    if check_shape(cx, (n, P), "cx", fn_label, check, issues):
        check_no_nan(cx, "cx", fn_label, check, issues)
    if check_shape(cxx, (n, n, P), "cxx", fn_label, check, issues):
        check_no_nan(cxx, "cxx", fn_label, check, issues)

    # fx/fu/cu/cxu/cuu: back_pass.py's loop only ever reads indices
    # 0..P-2 (see protocols.py's DerivsFn docstring: "last slice/column
    # unused") -- only check the columns it actually uses.
    last = slice(None, -1)
    if check_shape(fx, (n, n, P), "fx", fn_label, check, issues):
        check_no_nan(fx, "fx", fn_label, check, issues, axis_slice=last)
    if check_shape(fu, (n, m, P), "fu", fn_label, check, issues):
        check_no_nan(fu, "fu", fn_label, check, issues, axis_slice=last)
    if check_shape(cu, (m, P), "cu", fn_label, check, issues):
        check_no_nan(cu, "cu", fn_label, check, issues, axis_slice=last)
    if check_shape(cxu, (n, m, P), "cxu", fn_label, check, issues):
        check_no_nan(cxu, "cxu", fn_label, check, issues, axis_slice=last)
    if check_shape(cuu, (m, m, P), "cuu", fn_label, check, issues):
        check_no_nan(cuu, "cuu", fn_label, check, issues, axis_slice=last)

    # fxx/fxu/fuu: must be uniformly None (iLQG mode) or uniformly
    # provided (full DDP mode) -- see protocols.py's DerivsFn docstring.
    second_order = {"fxx": (fxx, (n, n, n, P)),
                     "fxu": (fxu, (n, n, m, P)),
                     "fuu": (fuu, (n, m, m, P))}
    provided = {k: v is not None for k, (v, _) in second_order.items()}
    if len(set(provided.values())) != 1:
        which = ", ".join(f"{k}={'provided' if v else 'None'}" for k, v in provided.items())
        issues.append(ValidationIssue(
            fn_label, check,
            f"fxx/fxu/fuu must be either all None (iLQG mode) or all provided "
            f"(full DDP mode), never a mix -- got {which}"))
    else:
        for k, (val, shape) in second_order.items():
            if val is not None and check_shape(val, shape, k, fn_label, check, issues):
                check_no_nan(val, k, fn_label, check, issues, axis_slice=last)

    return issues


def validate_constraint_fn(constraint_fn, *, n: int, m: int, N: int,
                              rng: np.random.Generator = None) -> list:
    """Validates against protocols.ConstraintFn (Dastan & Sensinger extension)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    issues: list = []
    fn_label = "constraint_fn"
    if not check_arity(constraint_fn, ["x_traj", "u_traj"], fn_label, issues):
        return issues

    check = "call"
    x_traj = rng.standard_normal((n, N))
    u_traj = rng.standard_normal((m, N))
    result = safe_call(constraint_fn, (x_traj, u_traj), fn_label, check, issues)
    if result is None:
        return issues
    if check_shape(result, (N, m, 2), "lims", fn_label, check, issues):
        lower, upper = result[..., 0], result[..., 1]
        bad = np.argwhere(lower > upper)
        if len(bad) > 0:
            i, j = bad[0]
            issues.append(ValidationIssue(
                fn_label, check,
                f"lower bound > upper bound at {len(bad)} (timestep, control) "
                f"position(s), e.g. lims[{i},{j}] -- the per-timestep shape has "
                f"no 'unconstrained' sentinel (unlike the static (m,2) case) "
                f"and box_qp.py will silently misbehave on an inverted box "
                f"rather than raising"))
    return issues


def validate_core_plugin(step_fn, derivs_fn, *, n: int, m: int, N: int,
                            constraint_fn=None, K: int = 3,
                            rng: np.random.Generator = None) -> None:
    """
    Runs every applicable check across step_fn/derivs_fn[/constraint_fn]
    and raises ONE PluginContractError listing every issue found, or
    returns silently (no output) if the plug-in is fully
    contract-compliant.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    issues = []
    issues += validate_step_fn(step_fn, n=n, m=m, N=N, K=K, rng=rng)
    issues += validate_derivs_fn(derivs_fn, n=n, m=m, N=N, rng=rng)
    if constraint_fn is not None:
        issues += validate_constraint_fn(constraint_fn, n=n, m=m, N=N, rng=rng)
    if issues:
        raise PluginContractError(issues)
