"""
Tests for wizard/constraint_diagnostics.py -- the "is the constraint
actually binding, or just coincidentally satisfied?" report shown on the
Run & Results page (see CLAUDE.md's "validating constraint satisfaction"
discussion).
"""

import numpy as np
import pytest

from wizard.constraint_diagnostics import compute_binding_report, summarize_binding
from wizard.core_ilqr_adapter import compile_ilqr, solve_ilqr
from wizard.schema import (
    ActionSpec, ConstraintSpec, MeasurementSpec, ModelConfig, StateSpec,
)
from wizard.test_robot_arm_example import _robot_arm_ilqr_config


def test_returns_none_when_no_constraints_are_configured():
    config = ModelConfig(
        name="no_constraints", dt=1.0, n_sessions=5,
        states=[StateSpec(name="x", initial_value=0.0)],
        actions=[ActionSpec(name="u", min=-1.0, max=1.0)],
        dynamics={"x": "x + u"},
        measurement=[MeasurementSpec(name="x", expression="x")],
    )
    result = {"x_true": np.zeros((1, 5)), "u": np.zeros((1, 4))}
    assert compute_binding_report(config, result) is None


def test_returns_none_for_a_malformed_result():
    config = _robot_arm_ilqr_config()
    assert compute_binding_report(config, {}) is None
    assert compute_binding_report(config, {"x_true": None, "u": None}) is None
    assert compute_binding_report(
        config, {"x_true": np.zeros((4, 5)), "u": np.zeros((2, 0))}) is None


def test_reports_binding_on_the_shipped_robot_arm_constraint():
    """End-to-end: solve the real shipped example, then confirm the
    diagnostic itself (not just the raw trajectory, as
    test_robot_arm_example.py's own test already checks) reports the
    constraint as genuinely binding on tau1 and untouched on tau2."""
    config = _robot_arm_ilqr_config()
    result = solve_ilqr(compile_ilqr(config))

    report = compute_binding_report(config, result)
    assert report is not None
    assert report.lims.shape == (result["u"].shape[1], config.nu, 2)

    stats_by_name = {s.action_name: s for s in report.per_action}
    tau1_stat = stats_by_name["tau1"]
    assert tau1_stat.n_tightened > 0
    assert tau1_stat.fraction_tightened == pytest.approx(1.0)
    assert tau1_stat.min_slack is not None
    assert tau1_stat.min_slack < 0.5  # genuinely close to its own bound early on

    tau2_stat = stats_by_name["tau2"]
    assert tau2_stat.n_tightened == 0  # the constraint never touches tau2
    assert tau2_stat.min_slack is None

    lines = summarize_binding(report)
    assert any("tau1" in line and "binding limit" in line for line in lines)
    assert any("tau2" in line and "never" in line for line in lines)


def test_reports_never_tightened_when_the_constraint_is_vacuous():
    """A cap far above the plain action bound can never tighten anything
    -- the diagnostic must say so plainly rather than fabricate binding
    evidence that isn't there."""
    config = _robot_arm_ilqr_config()
    config.constraints = [ConstraintSpec(
        name="vacuous", kind="state_action",
        expression="(1000 - 0.5*w1^2) - tau1", enabled=True)]
    result = solve_ilqr(compile_ilqr(config))

    report = compute_binding_report(config, result)
    assert report is not None
    tau1_stat = next(s for s in report.per_action if s.action_name == "tau1")
    assert tau1_stat.n_tightened == 0
    assert tau1_stat.min_slack is None

    lines = summarize_binding(report)
    assert any("never" in line and "no visible effect" in line for line in lines)


def test_returns_none_when_the_only_constraint_is_disabled():
    config = _robot_arm_ilqr_config()
    config.constraints[0].enabled = False
    result = solve_ilqr(compile_ilqr(config))
    assert compute_binding_report(config, result) is None
