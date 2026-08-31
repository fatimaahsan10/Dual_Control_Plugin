from wizard.schema import (
    ModelConfig, StateSpec, ActionSpec, ParameterSpec, ConstantSpec,
    MeasurementSpec, CostSpec,
)
from wizard.codegen_preview import (
    generate_constants_source, generate_continuous_dynamics_source,
    generate_measurement_source, generate_cost_source, generate_all_previews,
)


def _config() -> ModelConfig:
    return ModelConfig(
        name="preview_model",
        states=[StateSpec(name="x1", label="Demand")],
        actions=[ActionSpec(name="u1", min=0.0, max=10.0, label="Price")],
        parameters=[ParameterSpec(name="p1", prior_guess=0.5, label="Sensitivity")],
        constants=[ConstantSpec(name="a", value=100.0)],
        dynamics={"x1": "a - p1*u1 - x1"},
        measurement=[MeasurementSpec(name="y1", expression="x1", noise_scale=1.0)],
        cost=CostSpec(running="-(u1)*(a - p1*u1)", terminal="0"),
    )


def test_generate_all_previews_returns_expected_files():
    previews = generate_all_previews(_config())
    assert set(previews) == {"constants.py", "continuous_dynamics.py",
                               "measurement.py", "cost.py"}
    for text in previews.values():
        assert isinstance(text, str) and len(text) > 0


def test_constants_source_includes_prior_and_known_constants():
    src = generate_constants_source(_config())
    assert "0.5" in src  # p1's prior_guess
    assert "100.0" in src  # constant a's value
    assert "p1" in src
    assert "a" in src


def test_continuous_dynamics_source_includes_equation_and_names():
    src = generate_continuous_dynamics_source(_config())
    assert "x1" in src
    assert "u1" in src
    assert "p1" in src
    assert "def continuous_dynamics" in src


def test_measurement_source_includes_expression_and_noise_scale():
    src = generate_measurement_source(_config())
    assert "y1" in src
    assert "1.0" in src
    assert "def measurement" in src


def test_cost_source_includes_running_and_terminal():
    src = generate_cost_source(_config())
    assert "running cost" in src
    assert "terminal cost" in src
    assert "def cost" in src


def test_preview_never_raises_on_unparseable_equation():
    config = _config()
    config.dynamics["x1"] = "totally_unknown_name + a"
    src = generate_continuous_dynamics_source(config)
    assert "does not currently parse" in src
    assert "totally_unknown_name" in src


def test_preview_never_raises_on_empty_equation():
    config = _config()
    config.dynamics["x1"] = ""
    src = generate_continuous_dynamics_source(config)
    assert isinstance(src, str)


def test_preview_never_raises_on_structurally_incomplete_config():
    # A config mid-construction in the wizard (e.g. states defined but no
    # actions yet) must still render something, not crash the UI's live
    # preview panel.
    config = ModelConfig(name="incomplete")
    for fn in (generate_constants_source, generate_continuous_dynamics_source,
                generate_measurement_source, generate_cost_source):
        src = fn(config)
        assert isinstance(src, str)


def test_generate_all_previews_never_raises_on_incomplete_config():
    config = ModelConfig(name="incomplete")
    previews = generate_all_previews(config)
    assert len(previews) == 4
