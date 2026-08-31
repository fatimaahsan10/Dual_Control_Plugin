"""
Runs the wizard's full "Test This Model" step: compile the config into
callables (generic_plant.py), then check them against
extensions/dual_control/protocols.py's contract using the REAL,
UNMODIFIED extensions/dual_control/validate_plugin.validate_dual_control_plugin()
-- the same validator every hand-written plant in this codebase is
checked against, not a wizard-specific stand-in.

Two failure layers are translated into plain language here:
  1. PlantCompilationError (generic_plant.py) -- structural config
     problems or an equation that failed to parse. Its messages are
     already plain-language (schema.py/equation_parser.py write them that
     way), so they're passed through as-is.
  2. PluginContractError (core/ddp_solver/plugin_validation.py, raised by
     validate_dual_control_plugin) -- a shape/arity/NaN-hygiene problem
     found by calling the compiled callables with synthetic data. Its
     ValidationIssue list is reformatted with a plain-language label for
     which part of the model each issue came from.

A final broad `except Exception` backstop in each stage exists for
whatever neither layer anticipated (e.g. an equation that parses and
passes every documented check but still does something like divide by a
runtime-computed zero only for certain sampled inputs, raising
FloatingPointError under `numpy.errstate(raise=...)`, or any other
exception validate_dual_control_plugin's own `safe_call` wrapper doesn't
already convert to a ValidationIssue). This module NEVER raises --
run_validation() always returns a ValidationReport, so a caller (the
Streamlit UI) never needs its own try/except around this call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.ddp_solver.plugin_validation import PluginContractError, ValidationIssue
from extensions.dual_control.validate_plugin import validate_dual_control_plugin

from wizard.generic_plant import compile_plant, PlantCompilationError
from wizard.schema import ModelConfig

_FN_LABELS = {
    "dynamics": "the dynamics equations",
    "continuous_dynamics": "the dynamics equations",
    "measurement": "the measurement expressions",
    "cost": "the cost expression",
}


@dataclass
class ValidationReport:
    ok: bool
    messages: list[str] = field(default_factory=list)
    raw_error: str | None = None  # technical detail for a collapsed panel, not the headline


def _translate_issue(issue: ValidationIssue) -> str:
    label = _FN_LABELS.get(issue.fn, issue.fn)
    return f"Problem found in {label} ({issue.check}): {issue.message}"


def run_validation(config: ModelConfig) -> ValidationReport:
    """
    Never raises. Returns a ValidationReport whose `.ok` gates whether the
    wizard's "Run Simulation" step is allowed to proceed.
    """
    try:
        plant = compile_plant(config)
    except PlantCompilationError as e:
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

    return ValidationReport(ok=True)
