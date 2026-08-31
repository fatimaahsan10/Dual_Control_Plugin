"""
Shared plug-in validation infrastructure, reused by both
core/ddp_solver/validate_plugin.py and
extensions/dual_control/validate_plugin.py -- same cross-layer reuse
pattern as extensions/dual_control/backward_pass.py importing
core/ddp_solver/box_qp.py directly (a precedent already established in
this codebase, not a new layering decision).

Pure utility, no dynamics/cost/estimator math: the building blocks a
per-layer validator uses to (1) check a candidate callable's arity
without calling it, (2) call it with synthetic data and catch any
exception as a reportable issue instead of letting it propagate, and (3)
check the result's shape/tuple-length/NaN-hygiene against what its
Protocol documents -- each producing a specific ValidationIssue instead
of a bare AssertionError or a confusing traceback from three functions
away.
"""

import inspect
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ValidationIssue:
    fn: str      # e.g. "step_fn"
    check: str   # e.g. "terminal step (i=N, u=NaN)"
    message: str

    def __str__(self) -> str:
        return f"[{self.fn}] {self.check}: {self.message}"


class PluginContractError(Exception):
    """
    Raised by validate_core_plugin()/validate_dual_control_plugin() when
    one or more callables violate their documented Protocol contract.
    Carries EVERY issue found across every callable checked (not just
    the first one hit) -- see .issues -- so a plug-in author gets one
    report covering every mistake instead of fixing them one at a time
    across repeated runs.
    """

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = list(issues)
        header = f"{len(self.issues)} plug-in contract violation(s) found:"
        body = "\n".join(f"  - {i}" for i in self.issues)
        super().__init__(f"{header}\n{body}")


def check_arity(fn, param_names: Sequence[str], fn_label: str,
                 issues: list) -> bool:
    """
    Confirms `fn` accepts len(param_names) positional arguments, WITHOUT
    calling it -- so a plain arity mismatch is reported as one clear
    message ("does not accept N positional arguments (...)") instead of
    surfacing as a TypeError raised from deep inside a real call
    attempt, mixed in with whatever check happened to trigger it.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        issues.append(ValidationIssue(fn_label, "signature",
                                        f"could not introspect a signature ({e})"))
        return False
    try:
        sig.bind(*([None] * len(param_names)))
    except TypeError as e:
        issues.append(ValidationIssue(
            fn_label, "signature",
            f"does not accept {len(param_names)} positional arguments "
            f"({', '.join(param_names)}) as documented in its Protocol -- "
            f"actual signature is {sig}: {e}"))
        return False
    return True


def safe_call(fn, args: Sequence, fn_label: str, check: str, issues: list):
    """
    Calls fn(*args), converting any exception into a ValidationIssue
    (exception type + message preserved) instead of letting it propagate
    and abort every other check that hasn't run yet. Returns None on
    failure -- callers must guard on that before inspecting a result.
    """
    try:
        return fn(*args)
    except Exception as e:
        issues.append(ValidationIssue(fn_label, check,
                                        f"raised {type(e).__name__} during call: {e}"))
        return None


def check_return_tuple(result, expected_len: int, names: Sequence[str],
                         fn_label: str, check: str, issues: list) -> bool:
    if not (isinstance(result, tuple) and len(result) == expected_len):
        got = (f"a {type(result).__name__} of length {len(result)}"
               if isinstance(result, tuple) else f"a {type(result).__name__}")
        issues.append(ValidationIssue(
            fn_label, check,
            f"expected a {expected_len}-tuple ({', '.join(names)}), got {got}"))
        return False
    return True


def check_shape(value, expected_shape: tuple, label: str, fn_label: str,
                 check: str, issues: list, allow_none: bool = False) -> bool:
    if value is None:
        if allow_none:
            return True
        issues.append(ValidationIssue(
            fn_label, check,
            f"{label}: expected an ndarray of shape {expected_shape}, got None"))
        return False
    if not isinstance(value, np.ndarray):
        issues.append(ValidationIssue(
            fn_label, check, f"{label}: expected an ndarray, got {type(value).__name__}"))
        return False
    if value.shape != expected_shape:
        issues.append(ValidationIssue(
            fn_label, check,
            f"{label}: expected shape {expected_shape}, got {value.shape}"))
        return False
    return True


def check_no_nan(value, label: str, fn_label: str, check: str, issues: list,
                  axis_slice=None) -> bool:
    """
    axis_slice, if given, is applied to the LAST axis before checking
    (e.g. slice(None, -1) to exclude a documented-unused terminal
    column/slice from the check -- see e.g. back_pass.py never reading
    fx/fu/cu/cxu/cuu's last column).
    """
    if not isinstance(value, np.ndarray):
        return True  # a shape-check issue was already reported for this
    v = value[..., axis_slice] if axis_slice is not None else value
    if np.isnan(v).any():
        issues.append(ValidationIssue(
            fn_label, check, f"{label} contains NaN for finite, in-range inputs"))
        return False
    return True
