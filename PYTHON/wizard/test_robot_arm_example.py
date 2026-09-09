"""
Tests for wizard/examples/robot_arm_ilqr_constrained.json -- the shipped
"generic Wizard, one example model" demonstration (see Plan B discussion):
the 2-link robot arm's manipulator-equation dynamics, transcribed into the
wizard's own generic equation schema, run under control_method="ilqr" with
one "state_action" constraint (a state-dependent torque derating) built
through wizard/core_ilqr_adapter.py's composition of the existing,
UNMODIFIED extensions/constraints/ machinery.

Same "prove the generic path reproduces a proven-correct hand-written
reference exactly" convention test_generic_plant.py already uses for
dynamic_pricing/duopoly_pricing -- here cross-checked against
applications/robot_arm_2link/continuous_dynamics.py directly.
"""

import numpy as np
import pytest

from applications.robot_arm_2link.arm_constants import RobotArmConstants
from applications.robot_arm_2link import continuous_dynamics as real_arm_cd

from wizard.core_ilqr_adapter import compile_ilqr, solve_ilqr
from wizard.generic_plant import compile_plant
from wizard.schema import (
    ActionSpec, ConstantSpec, ConstraintSpec, CostSpec, MeasurementSpec,
    ModelConfig, StateSpec,
)
from wizard.solver_runner import run_solver
from wizard.storage import list_examples, load_model_from_path
from wizard.validation_runner import run_validation

_M2 = 1.0  # known link-2 mass -- matches applications/robot_arm_2link's M2_TRUE

_DYN_W1 = (
    "((m2*lc2^2 + I2)*(tau1 + m2*l1*lc2*sin(q2)*w1*w2 + m2*l1*lc2*sin(q2)*(w1 + w2)*w2 "
    "- ((m1*lc1 + m2*l1)*g*cos(q1) + m2*lc2*g*cos(q1 + q2)) - fv1*w1) "
    "- (m2*(lc2^2 + l1*lc2*cos(q2)) + I2)*(tau2 - m2*l1*lc2*sin(q2)*w1^2 "
    "- m2*lc2*g*cos(q1 + q2) - fv2*w2)) "
    "/ ((m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2)) + I1 + I2)*(m2*lc2^2 + I2) "
    "- (m2*(lc2^2 + l1*lc2*cos(q2)) + I2)^2)"
)
_DYN_W2 = (
    "(-(m2*(lc2^2 + l1*lc2*cos(q2)) + I2)*(tau1 + m2*l1*lc2*sin(q2)*w1*w2 "
    "+ m2*l1*lc2*sin(q2)*(w1 + w2)*w2 - ((m1*lc1 + m2*l1)*g*cos(q1) "
    "+ m2*lc2*g*cos(q1 + q2)) - fv1*w1) "
    "+ (m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2)) + I1 + I2)*(tau2 "
    "- m2*l1*lc2*sin(q2)*w1^2 - m2*lc2*g*cos(q1 + q2) - fv2*w2)) "
    "/ ((m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2)) + I1 + I2)*(m2*lc2^2 + I2) "
    "- (m2*(lc2^2 + l1*lc2*cos(q2)) + I2)^2)"
)


def _robot_arm_ilqr_config() -> ModelConfig:
    """Reproduces the shipped example's math by hand (rather than loading
    the JSON) so this test doesn't depend on the file surviving future
    edits unnoticed -- test_shipped_example_file_matches_this_config below
    is what actually guards the JSON itself."""
    return ModelConfig(
        name="robot_arm_ilqr", dt=0.05, n_sessions=20, control_method="ilqr",
        states=[StateSpec(name="q1", initial_value=0.0),
                 StateSpec(name="q2", initial_value=0.0),
                 StateSpec(name="w1", initial_value=0.0),
                 StateSpec(name="w2", initial_value=0.0)],
        actions=[ActionSpec(name="tau1", min=-30.0, max=30.0),
                  ActionSpec(name="tau2", min=-15.0, max=15.0)],
        dynamics={"q1": "w1", "q2": "w2", "w1": _DYN_W1, "w2": _DYN_W2},
        measurement=[MeasurementSpec(name="q1", expression="q1"),
                      MeasurementSpec(name="q2", expression="q2"),
                      MeasurementSpec(name="w1", expression="w1"),
                      MeasurementSpec(name="w2", expression="w2")],
        cost=CostSpec(
            running="10*(q1-1.0472)^2 + 10*(q2+0.7854)^2 + w1^2 + w2^2 "
                     "+ 0.01*tau1^2 + 0.01*tau2^2",
            terminal="200*(q1-1.0472)^2 + 200*(q2+0.7854)^2 + 50*w1^2 + 50*w2^2"),
        constants=[ConstantSpec(name="l1", value=1.0), ConstantSpec(name="l2", value=1.0),
                    ConstantSpec(name="lc1", value=0.5), ConstantSpec(name="lc2", value=0.5),
                    ConstantSpec(name="m1", value=1.0), ConstantSpec(name="I1", value=0.1),
                    ConstantSpec(name="I2", value=0.1), ConstantSpec(name="g", value=9.81),
                    ConstantSpec(name="fv1", value=0.5), ConstantSpec(name="fv2", value=0.5),
                    ConstantSpec(name="m2", value=_M2)],
        constraints=[ConstraintSpec(name="tau1_speed_derate", kind="state_action",
                                      expression="(28 - 0.5*w1^2) - tau1", enabled=True)],
    )


def _real_arm_constants() -> RobotArmConstants:
    return RobotArmConstants(
        l1=1.0, l2=1.0, lc1=0.5, lc2=0.5, m1=1.0, I1=0.1, I2=0.1, g=9.81,
        fv1=0.5, fv2=0.5, m2_prior=_M2, q1_target=np.pi / 3, q2_target=-np.pi / 4)


def test_robot_arm_example_is_shipped_and_discoverable():
    examples = list_examples()
    matches = [p for p in examples if "robot_arm_ilqr" in p.stem]
    assert len(matches) == 1


def test_shipped_example_file_matches_this_test_files_config():
    """Guards against the JSON drifting silently out of sync with the
    math this test file cross-checks -- loads the ACTUAL shipped file
    and confirms it's identical to _robot_arm_ilqr_config()."""
    examples = list_examples()
    path = next(p for p in examples if "robot_arm_ilqr" in p.stem)
    loaded = load_model_from_path(path)
    hand_built = _robot_arm_ilqr_config()
    assert loaded.control_method == hand_built.control_method
    assert loaded.dynamics == hand_built.dynamics
    assert loaded.cost == hand_built.cost
    assert [(c.name, c.value) for c in loaded.constants] == \
        [(c.name, c.value) for c in hand_built.constants]
    assert [(c.name, c.kind, c.expression, c.enabled) for c in loaded.constraints] == \
        [(c.name, c.kind, c.expression, c.enabled) for c in hand_built.constraints]


def test_continuous_dynamics_matches_real_robot_arm_module():
    config = _robot_arm_ilqr_config()
    plant = compile_plant(config)
    real_constants = _real_arm_constants()

    rng = np.random.default_rng(0)
    K = 6
    q = rng.uniform(-0.5, 0.5, size=(2, K))
    w = rng.uniform(-2.0, 2.0, size=(2, K))
    xa = np.vstack([q, w])
    u = np.vstack([rng.uniform(-30, 30, size=K), rng.uniform(-15, 15, size=K)])

    wiz_xdot, *_ = plant.continuous_dynamics(0.05, xa, u, plant.constants, False, None, 1)
    real_xdot, *_ = real_arm_cd.continuous_dynamics(0.05, xa, u, real_constants, False, None, 1)

    np.testing.assert_allclose(wiz_xdot, real_xdot, atol=1e-10)


def test_robot_arm_example_validates_and_runs_end_to_end():
    config = _robot_arm_ilqr_config()
    vreport = run_validation(config)
    assert vreport.ok, vreport.messages
    assert vreport.warnings == []

    sreport = run_solver(config)
    assert sreport.ok, sreport.messages
    result = sreport.result
    assert result["stop_reason"].startswith("SUCCESS")

    q1_final, q2_final = result["x_true"][0, -1], result["x_true"][1, -1]
    assert q1_final == pytest.approx(np.pi / 3, abs=0.3)   # ~60 deg target
    assert q2_final == pytest.approx(-np.pi / 4, abs=0.1)  # ~-45 deg target


def test_torque_derating_constraint_actually_binds_early():
    """The whole point of shipping this constraint: joint-1 torque must
    be capped by (28 - 0.5*w1^2), tighter than the plain action bound
    (tau1 in [-30, 30]), for at least the first few steps -- confirms the
    constraint is doing real, visible work, not passing vacuously."""
    config = _robot_arm_ilqr_config()
    result = solve_ilqr(compile_ilqr(config))
    tau1 = result["u"][0]
    w1 = result["x_true"][2, :-1]
    allowance = 28.0 - 0.5 * w1 ** 2

    # The box bound is linearized once per outer iteration around the
    # PREVIOUS accepted trajectory's w1, not re-solved against the final
    # trajectory's own w1 -- a small residual gap (~1e-3) between "the
    # bound as computed" and "the bound evaluated on the final w1" is
    # expected here (see core_ilqr_adapter.py's ConstraintFn composition
    # docstring), not evidence of a broken constraint.
    assert np.all(tau1 <= allowance + 1e-3)  # never meaningfully exceeds its own cap
    assert tau1[0] == pytest.approx(28.0, abs=0.05)  # binds exactly at the start (w1=0)
    assert tau1.max() < 30.0  # genuinely tighter than the plain action bound
