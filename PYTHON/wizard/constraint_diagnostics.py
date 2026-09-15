"""
Quantifies whether an enabled constraint (schema.ConstraintSpec) actually
bound a solved trajectory, or was only coincidentally satisfied -- see
CLAUDE.md's "validating constraint satisfaction" discussion. A trajectory
that stays far below its bound the whole run hasn't actually exercised the
constraint machinery (the same result would come out with the constraint
removed); only one that repeatedly gets pushed up against its own
(possibly state-dependent) bound is real evidence the constraint is
shaping the controller's choices, not just a plot someone has to eyeball.

Reuses wizard/core_ilqr_adapter.py's build_constraint_fn() -- the SAME
compiled constraint_fn both solver paths (wizard/solver_runner.py's "ilqg"
and "ilqr" branches) already run with -- purely to RE-EVALUATE it against
the already-solved trajectory's own (x, u). This module runs no solver of
its own and changes nothing about how either path already runs; it is a
read-only diagnostic over an existing result dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wizard.core_ilqr_adapter import ConstraintCompilationError, build_constraint_fn
from wizard.generic_plant import PlantCompilationError, compile_plant
from wizard.schema import ModelConfig

# An action is scored as "touching" its per-step bound when the gap between
# the value actually chosen and that bound is within this fraction of the
# action's own full [min, max] range -- scale-free, so it works whether an
# action's natural units are torques, prices, or anything else the wizard
# might be configured for.
_TOUCH_FRACTION = 0.02
_TIGHTEN_TOL = 1e-9


@dataclass
class ActionBindingStat:
    action_name: str
    n_total: int
    n_tightened: int
    fraction_tightened: float
    min_slack: Optional[float]        # None if never tightened
    fraction_touching: Optional[float]  # None if never tightened


@dataclass
class BindingReport:
    lims: np.ndarray  # (N, nu, 2) -- the actual per-step bound used, for
                        # plotting.plot_actions' optional overlay
    per_action: list[ActionBindingStat]


def compute_binding_report(config: ModelConfig, result: dict) -> Optional[BindingReport]:
    """Returns None whenever there's nothing meaningful to report --  no
    enabled constraints, a malformed/degenerate result, or the constraint
    fails to (re)compile -- never raises. Matches every other wizard
    report/plot function's "a failure here must not crash the results
    page" convention (see plotting.py's own module docstring)."""
    if not any(c.enabled for c in config.constraints):
        return None

    x_true = result.get("x_true")
    u = result.get("u")
    if (not isinstance(x_true, np.ndarray) or x_true.ndim != 2
            or not isinstance(u, np.ndarray) or u.ndim != 2 or u.shape[1] == 0):
        return None

    try:
        plant = compile_plant(config)
        constraint_fn = build_constraint_fn(plant, config)
    except (PlantCompilationError, ConstraintCompilationError):
        return None
    if constraint_fn is None:
        return None

    N = u.shape[1]
    x_for_u = x_true[:, :N]
    try:
        lims = constraint_fn(x_for_u, u)
    except Exception:  # noqa: BLE001 -- a diagnostic must never crash the results page
        return None
    if (not isinstance(lims, np.ndarray) or lims.shape != (N, config.nu, 2)
            or not np.isfinite(lims).all()):
        return None

    per_action = []
    for j, action in enumerate(config.actions):
        span = action.max - action.min
        touch_tol = max(span * _TOUCH_FRACTION, 1e-9)
        lo, hi = lims[:, j, 0], lims[:, j, 1]
        tightened = (lo > action.min + _TIGHTEN_TOL) | (hi < action.max - _TIGHTEN_TOL)
        n_tight = int(tightened.sum())
        if n_tight:
            uj = u[j, :]
            slack = np.minimum(uj - lo, hi - uj)[tightened]
            min_slack = float(slack.min())
            fraction_touching = float((slack < touch_tol).mean())
        else:
            min_slack = None
            fraction_touching = None
        per_action.append(ActionBindingStat(
            action_name=action.label or action.name, n_total=N,
            n_tightened=n_tight, fraction_tightened=n_tight / N,
            min_slack=min_slack, fraction_touching=fraction_touching))

    return BindingReport(lims=lims, per_action=per_action)


def summarize_binding(report: BindingReport) -> list[str]:
    """Plain-language lines, same convention as validation_runner.py's own
    ValidationReport messages -- meant to be shown directly in the UI."""
    lines: list[str] = []
    for s in report.per_action:
        if s.n_tightened == 0:
            lines.append(
                f"**{s.action_name}**: the constraint was never tighter "
                f"than its plain action bound on this run -- it had no "
                f"visible effect here, so this run alone doesn't "
                f"demonstrate that the constraint is being enforced.")
            continue
        pct = s.fraction_tightened * 100
        touch_pct = (s.fraction_touching or 0.0) * 100
        lines.append(
            f"**{s.action_name}**: the constraint was the binding limit "
            f"on {s.n_tightened}/{s.n_total} steps ({pct:.0f}%); the "
            f"chosen value came within a small margin of that limit on "
            f"{touch_pct:.0f}% of those steps (closest gap: "
            f"{s.min_slack:.4g}). This is direct evidence the system is "
            f"actively being held to the constraint, not just "
            f"coincidentally satisfying it.")
    return lines
