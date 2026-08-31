import numpy as np
from matplotlib.figure import Figure

from wizard.schema import (
    ModelConfig, StateSpec, ActionSpec, ParameterSpec, MeasurementSpec, CostSpec,
    SolverSettings,
)
from wizard.solver_runner import run_solver
from wizard.plotting import (
    plot_states, plot_actions, plot_parameters, plot_cost, results_table,
)


def _config_with_params() -> ModelConfig:
    return ModelConfig(
        name="plot_test_model",
        n_sessions=4,
        states=[StateSpec(name="x1", initial_value=50.0, min=0.0, label="Demand")],
        actions=[ActionSpec(name="u1", min=10.0, max=45.0, label="Price")],
        parameters=[ParameterSpec(name="p1", prior_guess=0.5, prior_variance=1.0,
                                    true_value=2.0, label="Sensitivity")],
        dynamics={"x1": "0.3 * (100 - p1*u1 - x1)"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="-(u1 - 10.0) * (100 - p1*u1)", terminal="0"),
        solver=SolverSettings(max_du_iterations=15, first_run_max_du_iterations=20),
    )


def _config_without_params() -> ModelConfig:
    return ModelConfig(
        name="plot_test_no_params",
        n_sessions=4,
        states=[StateSpec(name="x1", initial_value=1.0)],
        actions=[ActionSpec(name="u1", min=0.0, max=5.0)],
        dynamics={"x1": "u1 - x1"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="x1**2 + u1**2", terminal="0"),
        solver=SolverSettings(max_du_iterations=15, first_run_max_du_iterations=20),
    )


def test_all_plots_produced_for_a_real_successful_run():
    config = _config_with_params()
    report = run_solver(config)
    assert report.ok, report.messages

    fig_states = plot_states(report.result, config)
    fig_actions = plot_actions(report.result, config)
    fig_params = plot_parameters(report.result, config)
    fig_cost = plot_cost(report.result)

    assert isinstance(fig_states, Figure)
    assert isinstance(fig_actions, Figure)
    assert isinstance(fig_params, Figure)
    assert isinstance(fig_cost, Figure)


def test_results_table_has_expected_columns_and_length():
    config = _config_with_params()
    report = run_solver(config)
    assert report.ok, report.messages

    df = results_table(report.result, config)
    assert df is not None
    assert "x1 (true)" in df.columns
    assert "u1" in df.columns
    assert "p1 (estimated)" in df.columns
    assert len(df) == config.n_sessions


def test_plot_parameters_returns_none_when_no_unknown_parameters():
    config = _config_without_params()
    report = run_solver(config)
    assert report.ok, report.messages
    assert plot_parameters(report.result, config) is None


def test_states_and_actions_and_cost_still_plot_without_parameters():
    config = _config_without_params()
    report = run_solver(config)
    assert report.ok, report.messages
    assert isinstance(plot_states(report.result, config), Figure)
    assert isinstance(plot_actions(report.result, config), Figure)
    assert isinstance(plot_cost(report.result), Figure)


def test_empty_result_returns_none_not_a_crash():
    config = _config_with_params()
    assert plot_states({}, config) is None
    assert plot_actions({}, config) is None
    assert plot_parameters({}, config) is None
    assert plot_cost({}) is None
    assert results_table({}, config) is None


def test_all_nan_result_returns_none_not_a_crash():
    config = _config_with_params()
    bad_result = {
        "x_true": np.full((1, 4), np.nan),
        "u": np.full((1, 3), np.nan),
        "p_hat": np.full((1, 4), np.nan),
        "cost_true": np.full(4, np.nan),
    }
    assert plot_states(bad_result, config) is None
    assert plot_actions(bad_result, config) is None
    assert plot_parameters(bad_result, config) is None
    assert plot_cost(bad_result) is None


def test_plotting_never_raises_on_malformed_result_shapes():
    config = _config_with_params()
    malformed = {
        "x_true": np.array([1.0, 2.0, 3.0]),  # wrong ndim (1-D, not (nx, N))
        "u": "not an array",
        "p_hat": None,
        "cost_true": [1, 2, 3],  # plain list, not ndarray
    }
    # None of these should raise -- either produce a figure/table or None.
    for fn, args in (
            (plot_states, (malformed, config)),
            (plot_actions, (malformed, config)),
            (plot_parameters, (malformed, config)),
            (plot_cost, (malformed,)),
            (results_table, (malformed, config))):
        result = fn(*args)
        assert result is None or isinstance(result, (Figure, type(None)))
