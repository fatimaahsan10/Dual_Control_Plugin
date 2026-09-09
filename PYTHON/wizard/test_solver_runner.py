import time

import numpy as np
import pytest

import wizard.solver_runner as solver_runner_module
from wizard.schema import (
    ModelConfig, StateSpec, ActionSpec, ParameterSpec, ConstraintSpec,
    MeasurementSpec, CostSpec, SolverSettings,
)
from wizard.solver_runner import run_solver


def _small_config() -> ModelConfig:
    """A small, fast-to-run pricing-like model -- few sessions/iterations
    so these tests stay quick."""
    return ModelConfig(
        name="small_model",
        dt=1.0,
        n_sessions=3,
        states=[StateSpec(name="x1", initial_value=50.0, min=0.0)],
        actions=[ActionSpec(name="u1", min=10.0, max=45.0)],
        parameters=[ParameterSpec(name="p1", prior_guess=0.5, prior_variance=1.0,
                                    true_value=2.0)],
        constants=[],
        dynamics={"x1": "0.3 * (100 - p1*u1 - x1)"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="-(u1 - 10.0) * (100 - p1*u1)", terminal="0"),
        solver=SolverSettings(max_du_iterations=15, first_run_max_du_iterations=20),
    )


def test_valid_small_model_runs_end_to_end():
    report = run_solver(_small_config())
    assert report.ok, report.messages
    assert report.result is not None
    for key in ("x_hat", "p_hat", "x_true", "u", "cost_true", "total_true_cost"):
        assert key in report.result
    assert np.isfinite(report.result["total_true_cost"])
    assert report.elapsed_seconds >= 0.0


def test_zero_parameter_model_runs_end_to_end():
    config = ModelConfig(
        name="no_params_run",
        n_sessions=3,
        states=[StateSpec(name="x1", initial_value=1.0)],
        actions=[ActionSpec(name="u1", min=0.0, max=5.0)],
        dynamics={"x1": "u1 - x1"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="x1**2 + u1**2", terminal="0"),
        solver=SolverSettings(max_du_iterations=15, first_run_max_du_iterations=20),
    )
    report = run_solver(config)
    assert report.ok, report.messages


def test_structurally_invalid_config_never_reaches_the_solver():
    config = _small_config()
    config.actions = []
    report = run_solver(config)
    assert not report.ok
    assert report.result is None
    assert any("action" in m.lower() for m in report.messages)


def test_bad_equation_never_reaches_the_solver():
    config = _small_config()
    config.dynamics["x1"] = "p1*u1 - x1 + nonsense_name"
    report = run_solver(config)
    assert not report.ok
    assert any("nonsense_name" in m for m in report.messages)


def test_timeout_is_reported_plainly_and_does_not_raise(monkeypatch):
    def _slow_solver(*args, **kwargs):
        time.sleep(5)
        return {}

    monkeypatch.setattr(solver_runner_module, "main_outer_control_loop", _slow_solver)
    start = time.monotonic()
    report = run_solver(_small_config(), timeout_seconds=0.2)
    waited = time.monotonic() - start

    assert not report.ok
    assert any("longer than" in m for m in report.messages)
    assert waited < 4.0  # confirms run_solver returned promptly, not after the full sleep


def test_solver_exception_translated_to_plain_message(monkeypatch):
    def _raising_solver(*args, **kwargs):
        raise np.linalg.LinAlgError("Matrix is not positive definite")

    monkeypatch.setattr(solver_runner_module, "main_outer_control_loop", _raising_solver)
    report = run_solver(_small_config())
    assert not report.ok
    assert any("uncertainty" in m or "covariance" in m for m in report.messages)
    assert report.raw_error is not None
    assert "LinAlgError" in report.raw_error


def test_solver_valueerror_translated(monkeypatch):
    def _raising_solver(*args, **kwargs):
        raise ValueError("shapes do not align")

    monkeypatch.setattr(solver_runner_module, "main_outer_control_loop", _raising_solver)
    report = run_solver(_small_config())
    assert not report.ok
    assert any("shapes do not align" in m for m in report.messages)


def test_nan_result_is_caught_and_reported(monkeypatch):
    def _nan_solver(*args, **kwargs):
        return {"x_hat": np.array([[1.0, np.nan]]), "total_true_cost": 5.0}

    monkeypatch.setattr(solver_runner_module, "main_outer_control_loop", _nan_solver)
    report = run_solver(_small_config())
    assert not report.ok
    assert any("invalid numbers" in m for m in report.messages)
    assert any("x_hat" in m for m in report.messages)


def test_inf_result_is_caught_and_reported(monkeypatch):
    def _inf_solver(*args, **kwargs):
        return {"u": np.array([[1.0, np.inf]])}

    monkeypatch.setattr(solver_runner_module, "main_outer_control_loop", _inf_solver)
    report = run_solver(_small_config())
    assert not report.ok
    assert any("invalid numbers" in m for m in report.messages)


def test_run_solver_never_raises_on_garbage_config():
    config = ModelConfig(name="")
    report = run_solver(config)
    assert not report.ok
    assert isinstance(report.messages, list)
    assert len(report.messages) > 0


# ----------------------------------------------------------------------
# control_method == "ilqr" dispatch (Plan B: generic constraint layer)
# ----------------------------------------------------------------------

def _small_ilqr_config(constraints=None) -> ModelConfig:
    return ModelConfig(
        name="small_ilqr", dt=0.1, n_sessions=6, control_method="ilqr",
        states=[StateSpec(name="x1", initial_value=-2.0)],
        actions=[ActionSpec(name="u1", min=-10.0, max=10.0)],
        dynamics={"x1": "u1"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="x1^2 + 0.001*u1^2", terminal="5*x1^2"),
        constraints=constraints or [],
        solver=SolverSettings(first_run_max_du_iterations=30),
    )


def test_ilqr_control_method_runs_end_to_end():
    report = run_solver(_small_ilqr_config())
    assert report.ok, report.messages
    assert report.result is not None
    for key in ("x_true", "u", "cost_true", "total_true_cost", "trace", "stop_reason"):
        assert key in report.result
    assert "x_hat" not in report.result and "p_hat" not in report.result
    assert np.isfinite(report.result["total_true_cost"])


def test_ilqr_structurally_invalid_config_never_reaches_the_solver():
    config = _small_ilqr_config()
    config.actions = []
    report = run_solver(config)
    assert not report.ok
    assert report.result is None
    assert any("action" in m.lower() for m in report.messages)


def test_ilqr_bad_constraint_expression_never_reaches_the_solver():
    config = _small_ilqr_config(constraints=[
        ConstraintSpec(name="bad", kind="state_action", expression="1 - nonsense_name",
                        enabled=True)])
    report = run_solver(config)
    assert not report.ok
    assert any("nonsense_name" in m for m in report.messages)


def test_ilqr_timeout_is_reported_plainly_and_does_not_raise(monkeypatch):
    def _slow_solve(*args, **kwargs):
        time.sleep(5)
        return {}

    monkeypatch.setattr(solver_runner_module, "solve_ilqr", _slow_solve)
    start = time.monotonic()
    report = run_solver(_small_ilqr_config(), timeout_seconds=0.2)
    waited = time.monotonic() - start

    assert not report.ok
    assert any("longer than" in m for m in report.messages)
    assert waited < 4.0


def test_ilqr_solver_exception_translated_to_plain_message(monkeypatch):
    def _raising_solve(*args, **kwargs):
        raise ValueError("shapes do not align")

    monkeypatch.setattr(solver_runner_module, "solve_ilqr", _raising_solve)
    report = run_solver(_small_ilqr_config())
    assert not report.ok
    assert any("shapes do not align" in m for m in report.messages)


def test_ilqr_nan_result_is_caught_and_reported(monkeypatch):
    def _nan_solve(*args, **kwargs):
        return {"x_true": np.array([[1.0, np.nan]]), "total_true_cost": 5.0}

    monkeypatch.setattr(solver_runner_module, "solve_ilqr", _nan_solve)
    report = run_solver(_small_ilqr_config())
    assert not report.ok
    assert any("invalid numbers" in m for m in report.messages)
    assert any("x_true" in m for m in report.messages)
