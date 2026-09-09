"""
Runs the wizard's full "Test This Model" step. Dispatches on
`config.control_method` (see schema.py):

  "ilqg" (default) -- compile the config into callables
      (generic_plant.py), then check them against
      extensions/dual_control/protocols.py's contract using the REAL,
      UNMODIFIED extensions/dual_control/validate_plugin.
      validate_dual_control_plugin() -- the same validator every
      hand-written plant in this codebase is checked against, not a
      wizard-specific stand-in. PLUS, if the config has any enabled
      constraints, the same wizard/core_ilqr_adapter.py::
      build_constraint_fn() the "ilqr" path uses, checked with
      core/ddp_solver/validate_plugin.validate_constraint_fn() (that
      function's own contract is control-method-agnostic) and the same
      RuntimeWarning capture described below.

  "ilqr" -- compile the config via wizard/core_ilqr_adapter.py, then
      check the resulting step_fn/derivs_fn[/constraint_fn] against
      core/ddp_solver/protocols.py's contract using the REAL, UNMODIFIED
      core/ddp_solver/validate_plugin.validate_core_plugin().

Failure layers translated into plain language here:
  1. PlantCompilationError (generic_plant.py) -- structural config
     problems or a dynamics/measurement/cost equation that failed to
     parse. Its messages are already plain-language, passed through as-is.
  1b. ConstraintCompilationError (wizard/core_ilqr_adapter.py) -- same
     idea, for a constraint expression that failed to parse. Applies to
     BOTH control methods (see build_constraint_fn's own docstring on
     being reused by the "ilqg" path too).
  2. PluginContractError (core/ddp_solver/plugin_validation.py, raised by
     either validate_dual_control_plugin or validate_core_plugin) -- a
     shape/arity/NaN-hygiene problem found by calling the compiled
     callables with synthetic data. Its ValidationIssue list is
     reformatted with a plain-language label for which part of the model
     each issue came from.

CONSTRAINT-WARNING CAPTURE (both control methods): validate_constraint_fn
(core/ddp_solver/validate_plugin.py) exercises constraint_fn on synthetic
data but does not itself surface a captured Python `warnings.warn` (e.g.
extensions/constraints/dynamic_control_bounds.py's "relative degree is
not one here... silently unenforced" RuntimeWarning) as a ValidationIssue
-- left uncaptured, it would print to stderr and never reach the wizard
UI, silently leaving a user unaware a constraint they wrote isn't actually
constraining anything. Both _validate_ilqr and _validate_ilqg wrap the
relevant call in `warnings.catch_warnings(record=True)` and surface any
RuntimeWarning from extensions/constraints/ as a plain-language
ValidationReport message -- advisory (does not fail validation on its
own), since the constraint math itself is not wrong, only inapplicable at
the sampled points.

A final broad `except Exception` backstop in each stage exists for
whatever neither layer anticipated. This module NEVER raises --
run_validation() always returns a ValidationReport, so a caller (the
Streamlit UI) never needs its own try/except around this call.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from core.ddp_solver.plugin_validation import PluginContractError, ValidationIssue
from core.ddp_solver.validate_plugin import validate_constraint_fn, validate_core_plugin
from extensions.dual_control.validate_plugin import validate_dual_control_plugin

from wizard.core_ilqr_adapter import (
    build_constraint_fn, ConstraintCompilationError, compile_ilqr,
)
from wizard.generic_plant import compile_plant, PlantCompilationError
from wizard.schema import ModelConfig

_FN_LABELS = {
    "dynamics": "the dynamics equations",
    "continuous_dynamics": "the dynamics equations",
    "measurement": "the measurement expressions",
    "cost": "the cost expression",
    "step_fn": "the dynamics/cost equations",
    "derivs_fn": "the dynamics/cost equations",
    "constraint_fn": "the constraint(s)",
}


@dataclass
class ValidationReport:
    ok: bool
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # advisory, doesn't set ok=False
    raw_error: str | None = None  # technical detail for a collapsed panel, not the headline


def _translate_issue(issue: ValidationIssue) -> str:
    label = _FN_LABELS.get(issue.fn, issue.fn)
    return f"Problem found in {label} ({issue.check}): {issue.message}"


def _dedup_constraint_warnings(caught) -> list[str]:
    """De-duplicated by message text -- build_time_varying_lims calls
    solve_box_from_constraint once per trajectory timestep, so the SAME
    "relative degree is not one" warning can otherwise repeat dozens of
    times for one genuinely-unenforced constraint row. Shared by both
    _validate_ilqr and _validate_ilqg."""
    seen_texts: set[str] = set()
    messages = []
    for w in caught:
        if not issubclass(w.category, RuntimeWarning):
            continue
        text = str(w.message)
        if text in seen_texts:
            continue
        seen_texts.add(text)
        messages.append(
            f"One of your constraints has no effect at some sampled "
            f"points -- {text} This usually means the constraint (or its "
            f"Lie derivative, for a state-only constraint) doesn't "
            f"actually depend on any action at that point -- see the "
            f"relative-degree limitation described on the Constraints step.")
    return messages


def run_validation(config: ModelConfig) -> ValidationReport:
    """
    Never raises. Returns a ValidationReport whose `.ok` gates whether the
    wizard's "Run Simulation" step is allowed to proceed. Dispatches on
    config.control_method; see module docstring.
    """
    if config.control_method == "ilqr":
        return _validate_ilqr(config)
    return _validate_ilqg(config)


def _validate_ilqr(config: ModelConfig) -> ValidationReport:
    try:
        inputs = compile_ilqr(config)
    except (PlantCompilationError, ConstraintCompilationError) as e:
        return ValidationReport(ok=False, messages=list(e.errors))
    except Exception as e:  # noqa: BLE001 -- absolute backstop, see module docstring
        return ValidationReport(
            ok=False,
            messages=["Something unexpected went wrong while building "
                       "your model from its equations. See technical "
                       "details below, or try simplifying the equations."],
            raw_error=f"{type(e).__name__}: {e}")

    n, m = config.nx, config.nu
    N = config.n_sessions - 1
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_core_plugin(
                inputs.step_fn, inputs.derivs_fn, n=n, m=m, N=N,
                constraint_fn=inputs.constraint_fn)
    except PluginContractError as e:
        return ValidationReport(ok=False,
                                  messages=[_translate_issue(i) for i in e.issues])
    except Exception as e:  # noqa: BLE001 -- absolute backstop, see module docstring
        return ValidationReport(
            ok=False,
            messages=["An unexpected error occurred while testing your "
                       "model with sample numbers. This usually means an "
                       "equation produces something unexpected for some "
                       "inputs -- e.g. dividing by a value that can be "
                       "zero, or a very large exponent. See technical "
                       "details below."],
            raw_error=f"{type(e).__name__}: {e}")

    return ValidationReport(ok=True, warnings=_dedup_constraint_warnings(caught))


def _validate_ilqg(config: ModelConfig) -> ValidationReport:
    try:
        plant = compile_plant(config)
        constraint_fn = build_constraint_fn(plant, config)
    except (PlantCompilationError, ConstraintCompilationError) as e:
        return ValidationReport(ok=False, messages=list(e.errors))
    except Exception as e:  # noqa: BLE001 -- absolute backstop, see module docstring
        return ValidationReport(
            ok=False,
            messages=["Something unexpected went wrong while building "
                       "your model from its equations. See technical "
                       "details below, or try simplifying the equations."],
            raw_error=f"{type(e).__name__}: {e}")

    try:
        validate_dual_control_plugin(
            plant.dynamics, plant.measurement, plant.cost,
            plant.continuous_dynamics, nx=plant.nx, nu=plant.nu,
            nv=plant.ny, ny=plant.ny, constants=plant.constants,
            n_p=plant.n_p)

        caught = []
        if constraint_fn is not None:
            N = max(config.n_sessions - 1, 1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                issues = validate_constraint_fn(
                    constraint_fn, n=plant.nx, m=plant.nu, N=N)
            if issues:
                raise PluginContractError(issues)
    except PluginContractError as e:
        return ValidationReport(ok=False,
                                  messages=[_translate_issue(i) for i in e.issues])
    except Exception as e:  # noqa: BLE001 -- absolute backstop, see module docstring
        return ValidationReport(
            ok=False,
            messages=["An unexpected error occurred while testing your "
                       "model with sample numbers. This usually means an "
                       "equation produces something unexpected for some "
                       "inputs -- e.g. dividing by a value that can be "
                       "zero, or a very large exponent. See technical "
                       "details below."],
            raw_error=f"{type(e).__name__}: {e}")

    return ValidationReport(ok=True, warnings=_dedup_constraint_warnings(caught))
