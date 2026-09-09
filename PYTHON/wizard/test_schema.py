import pytest

from wizard.schema import (
    ModelConfig, StateSpec, ActionSpec, ParameterSpec, ConstantSpec,
    ConstraintSpec, MeasurementSpec, CostSpec, SolverSettings, MAX_N_SESSIONS,
    MAX_DU_ITERATIONS,
)


def _minimal_valid_config() -> ModelConfig:
    """A small but complete, valid config -- shape modeled loosely on
    dynamic_pricing/ (1 state, 1 action, 1 unknown parameter) since that's
    the simplest of the two proven real cases."""
    return ModelConfig(
        name="test_model",
        dt=1.0,
        n_sessions=5,
        states=[StateSpec(name="x1", initial_value=0.0)],
        actions=[ActionSpec(name="u1", min=0.0, max=100.0)],
        parameters=[ParameterSpec(name="p1", prior_guess=0.5, prior_variance=1.0)],
        constants=[ConstantSpec(name="a", value=100.0)],
        dynamics={"x1": "a - p1*u1 - x1"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="-(a - p1*u1)*u1", terminal="0"),
    )


def test_minimal_config_is_valid():
    cfg = _minimal_valid_config()
    assert cfg.validate() == []


def test_dims_properties():
    cfg = _minimal_valid_config()
    assert cfg.nx == 1
    assert cfg.nu == 1
    assert cfg.n_p == 1
    assert cfg.ny == 1


def test_json_round_trip():
    cfg = _minimal_valid_config()
    d = cfg.to_dict()
    restored = ModelConfig.from_dict(d)
    assert restored == cfg


def test_missing_name_flagged():
    cfg = _minimal_valid_config()
    cfg.name = ""
    errors = cfg.validate()
    assert any("name" in e.lower() for e in errors)


def test_no_states_flagged():
    cfg = _minimal_valid_config()
    cfg.states = []
    cfg.dynamics = {}
    errors = cfg.validate()
    assert any("at least one state" in e for e in errors)


def test_no_actions_flagged():
    cfg = _minimal_valid_config()
    cfg.actions = []
    errors = cfg.validate()
    assert any("at least one action" in e for e in errors)


def test_duplicate_name_across_categories_flagged():
    cfg = _minimal_valid_config()
    cfg.constants.append(ConstantSpec(name="x1", value=1.0))  # collides with state
    errors = cfg.validate()
    assert any("x1" in e and "unique" in e for e in errors)


def test_duplicate_name_within_category_flagged():
    cfg = _minimal_valid_config()
    cfg.constants.append(ConstantSpec(name="a", value=2.0))  # collides with itself
    errors = cfg.validate()
    assert any("more than once" in e for e in errors)


def test_reserved_function_name_flagged():
    cfg = _minimal_valid_config()
    cfg.constants.append(ConstantSpec(name="sin", value=1.0))
    errors = cfg.validate()
    assert any("reserved" in e for e in errors)


def test_invalid_identifier_flagged():
    cfg = _minimal_valid_config()
    cfg.states.append(StateSpec(name="2bad"))
    cfg.dynamics["2bad"] = "0"
    errors = cfg.validate()
    assert any("not a valid name" in e for e in errors)


def test_action_bounds_reversed_flagged():
    cfg = _minimal_valid_config()
    cfg.actions[0].min = 10.0
    cfg.actions[0].max = 5.0
    errors = cfg.validate()
    assert any("minimum" in e and "maximum" in e for e in errors)


def test_action_missing_bounds_flagged():
    cfg = _minimal_valid_config()
    cfg.actions[0].min = None  # type: ignore[assignment]
    errors = cfg.validate()
    assert any("numeric minimum" in e for e in errors)


def test_dynamics_missing_equation_flagged():
    cfg = _minimal_valid_config()
    cfg.dynamics = {}
    errors = cfg.validate()
    assert any("no equation" in e for e in errors)


def test_dynamics_extra_equation_flagged():
    cfg = _minimal_valid_config()
    cfg.dynamics["ghost"] = "0"
    errors = cfg.validate()
    assert any("ghost" in e and "isn't a defined state" in e for e in errors)


def test_empty_measurement_list_flagged():
    cfg = _minimal_valid_config()
    cfg.measurement = []
    errors = cfg.validate()
    assert any("at least one thing" in e for e in errors)


def test_duplicate_measurement_names_flagged():
    cfg = _minimal_valid_config()
    cfg.measurement.append(MeasurementSpec(name="y1", expression="x1"))
    errors = cfg.validate()
    assert any("share the same name" in e for e in errors)


def test_negative_noise_scale_flagged():
    cfg = _minimal_valid_config()
    cfg.measurement[0].noise_scale = -1.0
    errors = cfg.validate()
    assert any("can't be negative" in e for e in errors)


def test_empty_cost_flagged():
    cfg = _minimal_valid_config()
    cfg.cost.running = "   "
    errors = cfg.validate()
    assert any("running cost" in e for e in errors)


def test_n_sessions_hard_cap_enforced_even_if_hand_edited():
    cfg = _minimal_valid_config()
    cfg.n_sessions = MAX_N_SESSIONS + 1
    errors = cfg.validate()
    assert any("above the maximum" in e for e in errors)


def test_du_iterations_hard_cap_enforced():
    cfg = _minimal_valid_config()
    cfg.solver.max_du_iterations = MAX_DU_ITERATIONS + 1
    errors = cfg.validate()
    assert any("planning iterations per step" in e for e in errors)


def test_zero_parameters_is_valid_no_estimation_case():
    """n_p=0 (no unknown parameters) must be a legitimate, valid config --
    the wizard must not force every model to have something to estimate."""
    cfg = _minimal_valid_config()
    cfg.parameters = []
    cfg.dynamics["x1"] = "a - u1 - x1"  # no longer references p1
    cfg.cost.running = "-(a - u1)*u1"
    assert cfg.validate() == []
    assert cfg.n_p == 0


def test_validate_never_raises_on_garbage_config():
    """A hand-corrupted config must be safely reported, not crash the caller."""
    cfg = ModelConfig(name="")
    cfg.actions = [ActionSpec(name="1bad", min=None, max=None)]  # type: ignore
    cfg.states = [StateSpec(name="1bad")]  # duplicate/invalid name reused
    errors = cfg.validate()
    assert isinstance(errors, list)
    assert len(errors) > 0


# ----------------------------------------------------------------------
# control_method / ConstraintSpec (Plan B: generic constraint layer)
# ----------------------------------------------------------------------

def test_control_method_defaults_to_ilqg_for_backward_compatibility():
    """Every config saved before this field existed has no "control_method"
    key at all -- from_dict() must default it to "ilqg" so an old saved
    model keeps behaving exactly as before."""
    cfg = _minimal_valid_config()
    assert cfg.control_method == "ilqg"
    d = cfg.to_dict()
    del d["control_method"]
    restored = ModelConfig.from_dict(d)
    assert restored.control_method == "ilqg"


def test_invalid_control_method_flagged():
    cfg = _minimal_valid_config()
    cfg.control_method = "nonsense"
    errors = cfg.validate()
    assert any("Control method" in e for e in errors)


def test_ilqr_needs_at_least_two_sessions():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.n_sessions = 1
    errors = cfg.validate()
    assert any("at least 2 steps" in e for e in errors)


def test_constraints_default_empty_and_round_trip():
    cfg = _minimal_valid_config()
    assert cfg.constraints == []
    cfg.control_method = "ilqr"
    cfg.constraints = [ConstraintSpec(name="c1", kind="state_action",
                                        expression="1 - u1", enabled=True)]
    d = cfg.to_dict()
    restored = ModelConfig.from_dict(d)
    assert restored == cfg
    assert restored.constraints[0].kind == "state_action"


def test_constraint_missing_name_flagged():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.constraints = [ConstraintSpec(name="", kind="state_action", expression="1 - u1")]
    errors = cfg.validate()
    assert any("Every constraint needs a name" in e for e in errors)


def test_duplicate_constraint_name_flagged():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.constraints = [
        ConstraintSpec(name="c1", kind="state_action", expression="1 - u1"),
        ConstraintSpec(name="c1", kind="state_action", expression="2 - u1"),
    ]
    errors = cfg.validate()
    assert any("used more than once" in e for e in errors)


def test_invalid_constraint_kind_flagged():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.constraints = [ConstraintSpec(name="c1", kind="bogus", expression="1 - u1")]
    errors = cfg.validate()
    assert any("kind must be one of" in e for e in errors)


def test_empty_constraint_expression_flagged():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.constraints = [ConstraintSpec(name="c1", kind="state_action", expression="  ")]
    errors = cfg.validate()
    assert any("expression is empty" in e for e in errors)


def test_state_only_constraint_needs_positive_alpha():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.constraints = [ConstraintSpec(name="c1", kind="state_only",
                                        expression="1 - x1", alpha=0.0)]
    errors = cfg.validate()
    assert any("alpha must be a positive number" in e for e in errors)


def test_state_action_constraint_alpha_not_checked():
    """alpha is unused for kind=="state_action" -- an invalid value there
    must not be flagged (only state_only's Lie-derivative reduction uses
    it)."""
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.constraints = [ConstraintSpec(name="c1", kind="state_action",
                                        expression="1 - u1", alpha=0.0)]
    errors = cfg.validate()
    assert not any("alpha" in e for e in errors)


def test_enabled_constraint_requires_ilqr_control_method():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqg"  # default -- no constraint_fn hook there
    cfg.constraints = [ConstraintSpec(name="c1", kind="state_action",
                                        expression="1 - u1", enabled=True)]
    errors = cfg.validate()
    assert any("only enforced in iLQR" in e for e in errors)


def test_disabled_constraint_does_not_require_ilqr():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqg"
    cfg.constraints = [ConstraintSpec(name="c1", kind="state_action",
                                        expression="1 - u1", enabled=False)]
    errors = cfg.validate()
    assert not any("only enforced in iLQR" in e for e in errors)


def test_enabled_constraint_requires_box_qp_bound_method():
    cfg = _minimal_valid_config()
    cfg.control_method = "ilqr"
    cfg.n_sessions = 5
    cfg.solver.u_lim_method = 2  # tanh-squash
    cfg.constraints = [ConstraintSpec(name="c1", kind="state_action",
                                        expression="1 - u1", enabled=True)]
    errors = cfg.validate()
    assert any("Box-QP" in e for e in errors)
