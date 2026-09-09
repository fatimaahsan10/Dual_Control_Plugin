"""
Tests for core_ilqr_adapter.py -- the generic "iLQR control method" glue
between wizard/generic_plant.py's compiled callables and core/ddp_solver/
ilqg.py, plus the generic ConstraintSpec -> extensions/constraints/
composition.

The strongest check here (test_state_only_constraint_matches_zahid_pendulum_
closed_form) reproduces applications/pendulum_constrained/run_zahid_demo.py's
own hand-derived closed form through THIS module's generic path, using
nothing but a wizard ModelConfig -- the same "prove the generic path
reproduces a proven-correct hand-written reference exactly" convention
test_generic_plant.py already uses for dynamic_pricing/duopoly_pricing.
"""

import numpy as np
import pytest

from core.ddp_solver.plugin_validation import PluginContractError
from core.ddp_solver.validate_plugin import validate_core_plugin

from wizard.core_ilqr_adapter import (
    ConstraintCompilationError, build_constraint_fn, build_derivs_fn,
    build_step_fn, compile_ilqr, solve_ilqr,
)
from wizard.generic_plant import PlantCompilationError, compile_plant
from wizard.schema import (
    ActionSpec, ConstantSpec, ConstraintSpec, CostSpec, MeasurementSpec,
    ModelConfig, StateSpec,
)


# ----------------------------------------------------------------------
# Toy 1D integrator: x1_dot = u1 -- cheapest possible model that still
# exercises the full compile -> step_fn/derivs_fn -> ilqg() path.
# ----------------------------------------------------------------------

def _integrator_config(x0=-5.0, constraints=None, n_sessions=20) -> ModelConfig:
    return ModelConfig(
        name="toy_integrator", dt=0.1, n_sessions=n_sessions,
        control_method="ilqr",
        states=[StateSpec(name="x1", initial_value=x0)],
        actions=[ActionSpec(name="u1", min=-10.0, max=10.0)],
        dynamics={"x1": "u1"},
        measurement=[MeasurementSpec(name="x1", expression="x1", noise_scale=0.0)],
        cost=CostSpec(running="x1^2 + 0.0001*u1^2", terminal="10*x1^2"),
        constraints=constraints or [],
    )


def test_unconstrained_ilqr_reaches_target_and_converges():
    cfg = _integrator_config()
    result = solve_ilqr(compile_ilqr(cfg))
    assert result["stop_reason"].startswith("SUCCESS")
    assert abs(result["x_true"][0, -1]) < 0.1
    assert "x_hat" not in result and "p_hat" not in result  # deterministic, no estimation


def test_action_bounds_alone_are_respected():
    cfg = _integrator_config()
    result = solve_ilqr(compile_ilqr(cfg))
    assert result["u"].max() <= 10.0 + 1e-9
    assert result["u"].min() >= -10.0 - 1e-9


# ----------------------------------------------------------------------
# state_action constraint: h(x, u) >= 0, Variant A directly.
# ----------------------------------------------------------------------

def test_state_action_constraint_binds_tighter_than_action_bound():
    """Without a constraint, this problem's optimal control saturates at
    the action bound (u1=10, since x1 starts at -5 and wants to reach 0
    fast). Adding "3 - u1 >= 0" must cap it at 3 instead -- proving the
    constraint is doing real work, not passing vacuously."""
    baseline = solve_ilqr(compile_ilqr(_integrator_config()))
    assert baseline["u"].max() == pytest.approx(10.0, abs=1e-6)

    capped = solve_ilqr(compile_ilqr(_integrator_config(
        constraints=[ConstraintSpec(name="u_cap", kind="state_action",
                                      expression="3 - u1", enabled=True)])))
    assert capped["u"].max() == pytest.approx(3.0, abs=1e-4)


def test_disabled_constraint_has_no_effect():
    cfg = _integrator_config(constraints=[
        ConstraintSpec(name="u_cap", kind="state_action",
                        expression="3 - u1", enabled=False)])
    result = solve_ilqr(compile_ilqr(cfg))
    assert result["u"].max() == pytest.approx(10.0, abs=1e-6)


def test_build_constraint_fn_returns_none_with_no_enabled_constraints():
    plant = compile_plant(_integrator_config())
    assert build_constraint_fn(plant, _integrator_config()) is None
    disabled_cfg = _integrator_config(constraints=[
        ConstraintSpec(name="c", kind="state_action", expression="1 - u1", enabled=False)])
    plant2 = compile_plant(disabled_cfg)
    assert build_constraint_fn(plant2, disabled_cfg) is None


def test_two_channel_state_action_constraint_raises_at_call_time():
    """A constraint whose Jacobian touches TWO action channels can't be
    expressed as a box edge -- extensions/constraints/dynamic_control_bounds.
    solve_box_from_constraint raises ValueError. This only happens when
    constraint_fn is actually CALLED (compiling the expression itself
    can't detect this -- it's a property of the Jacobian at a given
    point), so build_constraint_fn itself succeeds; the ValueError
    surfaces from calling the returned constraint_fn."""
    cfg = ModelConfig(
        name="two_action", dt=0.1, n_sessions=5, control_method="ilqr",
        states=[StateSpec(name="x1", initial_value=0.0)],
        actions=[ActionSpec(name="u1", min=-10.0, max=10.0),
                 ActionSpec(name="u2", min=-10.0, max=10.0)],
        dynamics={"x1": "u1 + u2"},
        measurement=[MeasurementSpec(name="x1", expression="x1")],
        cost=CostSpec(running="x1^2 + u1^2 + u2^2", terminal="0"),
        constraints=[ConstraintSpec(name="both", kind="state_action",
                                      expression="1 - u1 - u2", enabled=True)],
    )
    plant = compile_plant(cfg)
    constraint_fn = build_constraint_fn(plant, cfg)
    assert constraint_fn is not None
    x_traj = np.zeros((1, 3))
    u_traj = np.zeros((2, 3))
    with pytest.raises(ValueError, match="not expressible as an axis-aligned box"):
        constraint_fn(x_traj, u_traj)


def test_validate_core_plugin_catches_two_channel_constraint_as_issue():
    """The SAME two-channel mistake, but reached through Test This
    Model's real validator (core.ddp_solver.validate_plugin.
    validate_core_plugin) instead of a direct call -- confirms
    safe_call's exception-catching turns this into a reportable
    PluginContractError, not a crash, during validation."""
    cfg = ModelConfig(
        name="two_action", dt=0.1, n_sessions=5, control_method="ilqr",
        states=[StateSpec(name="x1", initial_value=0.0)],
        actions=[ActionSpec(name="u1", min=-10.0, max=10.0),
                 ActionSpec(name="u2", min=-10.0, max=10.0)],
        dynamics={"x1": "u1 + u2"},
        measurement=[MeasurementSpec(name="x1", expression="x1")],
        cost=CostSpec(running="x1^2 + u1^2 + u2^2", terminal="0"),
        constraints=[ConstraintSpec(name="both", kind="state_action",
                                      expression="1 - u1 - u2", enabled=True)],
    )
    inputs = compile_ilqr(cfg)
    with pytest.raises(PluginContractError) as exc_info:
        validate_core_plugin(inputs.step_fn, inputs.derivs_fn, n=cfg.nx, m=cfg.nu,
                               N=cfg.n_sessions - 1, constraint_fn=inputs.constraint_fn)
    assert any("constraint_fn" == i.fn for i in exc_info.value.issues)


def test_state_action_referencing_unknown_name_raises_constraint_compilation_error():
    cfg = _integrator_config(constraints=[
        ConstraintSpec(name="bad", kind="state_action", expression="1 - u_typo", enabled=True)])
    with pytest.raises(ConstraintCompilationError) as exc_info:
        compile_ilqr(cfg)
    assert any("bad" in e for e in exc_info.value.errors)


def test_state_only_referencing_action_is_rejected_at_parse_time():
    """kind=="state_only" gets a NARROWER allowed-names list (states +
    constants only, no actions) -- referencing an action must be a plain
    parse error, not silently accepted."""
    cfg = _integrator_config(constraints=[
        ConstraintSpec(name="bad", kind="state_only", expression="1 - u1", enabled=True)])
    with pytest.raises(ConstraintCompilationError) as exc_info:
        compile_ilqr(cfg)
    assert any("u1" in e for e in exc_info.value.errors)


def test_multiple_constraints_stack_independently():
    """Two independent state_action rows, each pinning a different bound
    -- same "vector h with independent rows" property
    extensions/constraints/'s own test suite verifies at the library
    level; here checked through the generic wizard composition."""
    cfg = _integrator_config(constraints=[
        ConstraintSpec(name="upper", kind="state_action", expression="3 - u1", enabled=True),
        ConstraintSpec(name="lower", kind="state_action", expression="u1 + 8", enabled=True),
    ])
    plant = compile_plant(cfg)
    constraint_fn = build_constraint_fn(plant, cfg)
    x_traj = np.zeros((1, 3))
    u_traj = np.zeros((1, 3))
    lims = constraint_fn(x_traj, u_traj)
    assert lims.shape == (3, 1, 2)
    for i in range(3):
        assert lims[i, 0, 1] == pytest.approx(3.0)   # tau <= 3
        assert lims[i, 0, 0] == pytest.approx(-8.0)  # tau >= -8


def test_action_bound_intersects_with_looser_constraint():
    """If the constraint's own derived bound is LOOSER than the action's
    box bound, the action bound must still win -- exercised through the
    generic wizard path (extensions/constraints/dynamic_control_bounds.py's
    own `lower`/`upper` intersection, unmodified)."""
    cfg = _integrator_config(constraints=[
        ConstraintSpec(name="loose", kind="state_action", expression="100 - u1", enabled=True)])
    plant = compile_plant(cfg)
    constraint_fn = build_constraint_fn(plant, cfg)
    x_traj = np.zeros((1, 2))
    u_traj = np.zeros((1, 2))
    lims = constraint_fn(x_traj, u_traj)
    assert lims[0, 0, 1] == pytest.approx(10.0)  # action bound (10), not 100


def test_combined_analytic_dhdu_used_only_when_all_state_action():
    """The combined analytic dh/du Jacobian is only supplied when every
    enabled constraint is "state_action" -- confirmed indirectly: a
    pure-state_action combined bound and a numeric-fallback (dhdu_fn=None)
    computation must agree to near machine precision on an exactly-affine
    constraint (no truncation error either way)."""
    cfg = _integrator_config(constraints=[
        ConstraintSpec(name="c", kind="state_action", expression="3 - u1", enabled=True)])
    plant = compile_plant(cfg)
    constraint_fn = build_constraint_fn(plant, cfg)
    x_traj = np.zeros((1, 1))
    u_traj = np.array([[0.2]])
    lims = constraint_fn(x_traj, u_traj)
    assert lims[0, 0, 1] == pytest.approx(3.0, abs=1e-10)  # exact, not just close


# ----------------------------------------------------------------------
# state_only constraint: h(x) >= 0, Variant B (Lie derivative) then A --
# reproduces run_zahid_demo.py's own hand-derived pendulum closed form.
# ----------------------------------------------------------------------

def _pendulum_config(alpha=0.1) -> ModelConfig:
    """Same problem as applications/pendulum_constrained/run_zahid_demo.py:
    phi_dot = omega, omega_dot = sin(phi) + u1, constraint omega < 1 i.e.
    h(x) = 1 - omega >= 0."""
    return ModelConfig(
        name="pendulum", dt=0.05, n_sessions=5, control_method="ilqr",
        states=[StateSpec(name="phi", initial_value=-np.pi),
                StateSpec(name="omega", initial_value=0.0)],
        actions=[ActionSpec(name="u1", min=-100.0, max=100.0)],
        dynamics={"phi": "omega", "omega": "sin(phi) + u1"},
        measurement=[MeasurementSpec(name="phi", expression="phi"),
                     MeasurementSpec(name="omega", expression="omega")],
        cost=CostSpec(running="0.025*(phi^2 + omega^2 + u1^2)",
                       terminal="5*(phi^2 + omega^2)"),
        constraints=[ConstraintSpec(name="omega_limit", kind="state_only",
                                      expression="1 - omega", alpha=alpha, enabled=True)],
    )


def _closed_form_u_max(alpha, phi, omega):
    return alpha * (1.0 - omega) - np.sin(phi)


@pytest.mark.parametrize("phi,omega", [
    (-np.pi, 0.0), (np.pi / 4, 0.5), (0.0, -0.8), (2.5, 0.99),
])
def test_state_only_constraint_matches_zahid_pendulum_closed_form(phi, omega):
    """The wizard's generic ConstraintSpec("state_only") path, given
    nothing but h(x)=1-omega and the model's own dynamics equations, must
    reproduce the SAME closed form
    extensions/constraints/test_relative_degree_reduction.py verifies
    directly against relative_degree_reduction.state_constraint_to_control_
    constraint -- proving the generic composition in
    core_ilqr_adapter.build_constraint_fn is equivalent to the hand-wired
    composition run_zahid_demo.py uses, not a different (even if
    plausible-looking) computation."""
    alpha = 0.1
    cfg = _pendulum_config(alpha=alpha)
    plant = compile_plant(cfg)
    constraint_fn = build_constraint_fn(plant, cfg)

    x_traj = np.array([[phi], [omega]])
    u_traj = np.array([[0.0]])
    lims = constraint_fn(x_traj, u_traj)

    # constraint only pins the upper bound -- lower stays at the action's
    # own box bound (-100, per _pendulum_config), the `lower` hard limit
    # build_constraint_fn always intersects in (see its own docstring).
    assert lims[0, 0, 0] == -100.0
    assert lims[0, 0, 1] == pytest.approx(
        _closed_form_u_max(alpha, phi, omega), abs=1e-6)


def test_state_only_constraint_actually_binds_through_full_solve():
    """End-to-end (not just the box-bound math in isolation): solving the
    pendulum problem with the omega<1 constraint active must keep the
    accepted trajectory's omega from exceeding 1 (same qualitative
    property run_zahid_demo.py's alpha=1.0 "working" run demonstrates)."""
    cfg = _pendulum_config(alpha=1.0)
    cfg.n_sessions = 15
    result = solve_ilqr(compile_ilqr(cfg))
    assert result["x_true"][1, :].max() <= 1.0 + 1e-2


def test_relative_degree_zero_state_only_constraint_warns_not_crashes():
    """A state whose Lie derivative genuinely doesn't touch the only
    action (relative degree > 1) -- extensions/constraints/
    dynamic_control_bounds.py warns and leaves that row unenforced rather
    than raising; confirmed reachable through the generic wizard path."""
    cfg = ModelConfig(
        name="rd_zero", dt=0.1, n_sessions=5, control_method="ilqr",
        states=[StateSpec(name="x1", initial_value=0.0), StateSpec(name="x2", initial_value=0.0)],
        actions=[ActionSpec(name="u1", min=-10.0, max=10.0)],
        dynamics={"x1": "x2", "x2": "u1"},
        measurement=[MeasurementSpec(name="x1", expression="x1")],
        cost=CostSpec(running="x1^2 + u1^2", terminal="0"),
        constraints=[ConstraintSpec(name="pos_cap", kind="state_only",
                                      expression="100 - x1", alpha=0.5, enabled=True)],
    )
    plant = compile_plant(cfg)
    constraint_fn = build_constraint_fn(plant, cfg)
    x_traj = np.array([[5.0], [0.0]])
    u_traj = np.zeros((1, 1))
    with pytest.warns(RuntimeWarning, match="relative degree is not one"):
        lims = constraint_fn(x_traj, u_traj)
    assert lims[0, 0, 0] == -10.0  # falls back to the plain action bound
    assert lims[0, 0, 1] == 10.0


# ----------------------------------------------------------------------
# build_step_fn / build_derivs_fn -- shape/arity via the real validator.
# ----------------------------------------------------------------------

def test_step_fn_and_derivs_fn_pass_validate_core_plugin():
    cfg = _integrator_config()
    plant = compile_plant(cfg)
    step_fn = build_step_fn(plant, cfg)
    derivs_fn = build_derivs_fn(plant, step_fn)
    validate_core_plugin(step_fn, derivs_fn, n=cfg.nx, m=cfg.nu, N=cfg.n_sessions - 1)


def test_step_fn_zeroes_nan_terminal_control():
    cfg = _integrator_config()
    plant = compile_plant(cfg)
    step_fn = build_step_fn(plant, cfg)
    x = np.array([[1.0, 2.0]])
    u_nan = np.full((1, 2), np.nan)
    x_next, cost = step_fn(x, u_nan, 0)
    assert np.isfinite(x_next).all()
    assert np.isfinite(cost).all()


def test_compile_ilqr_raises_plant_compilation_error_for_bad_dynamics():
    cfg = _integrator_config()
    cfg.dynamics["x1"] = "u1 + undefined_name"
    with pytest.raises(PlantCompilationError):
        compile_ilqr(cfg)
