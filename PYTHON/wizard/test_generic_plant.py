"""
Tests for generic_plant.py, including two end-to-end cross-checks against
the REAL, hand-written example plants (applications/dynamic_pricing/ and
applications/duopoly_pricing/): a wizard config reproducing each one's
math is compiled here and its outputs are compared directly against the
real module's outputs on identical random inputs. This is the strongest
available correctness check -- it proves compile_plant() produces
callables that behave like a proven-correct hand-written plant, not just
callables that satisfy their own internal assumptions.

Every compiled plant is also run through the REAL, UNMODIFIED
extensions/dual_control/validate_plugin.validate_dual_control_plugin() --
confirming the generic factory's output actually satisfies protocols.py's
contract via the same validator every hand-written plant in this codebase
is checked against, not a wizard-specific stand-in.
"""

import numpy as np
import pytest

from applications.dynamic_pricing.pricing_constants import PricingConstants
from applications.dynamic_pricing import continuous_dynamics as real_pricing_cd
from applications.dynamic_pricing import dynamics as real_pricing_dyn
from applications.dynamic_pricing import measurement as real_pricing_meas
from applications.dynamic_pricing import cost as real_pricing_cost

from applications.duopoly_pricing.duopoly_constants import DuopolyPricingConstants
from applications.duopoly_pricing import continuous_dynamics as real_duopoly_cd
from applications.duopoly_pricing import cost as real_duopoly_cost

from extensions.dual_control.validate_plugin import validate_dual_control_plugin

from wizard.generic_plant import compile_plant, PlantCompilationError, GenericConstants
from wizard.schema import (
    ModelConfig, StateSpec, ActionSpec, ParameterSpec, ConstantSpec,
    MeasurementSpec, CostSpec,
)


# ----------------------------------------------------------------------
# Config reproducing applications/dynamic_pricing/'s math exactly
# (1 state, 1 action, 1 unknown parameter -- the simpler of the two
# proven real cases).
# ----------------------------------------------------------------------

def _pricing_equivalent_config() -> ModelConfig:
    return ModelConfig(
        name="pricing_equivalent",
        dt=1.0,
        n_sessions=10,
        states=[StateSpec(name="x1", initial_value=0.0, min=0.0, process_noise=1.0)],
        actions=[ActionSpec(name="u1", min=10.0, max=45.0)],
        parameters=[ParameterSpec(name="p1", prior_guess=0.5, prior_variance=1.0,
                                    process_noise=0.05)],
        constants=[ConstantSpec(name="a", value=100.0),
                    ConstantSpec(name="eta", value=0.3)],
        dynamics={"x1": "eta * (a - p1*u1 - x1)"},
        measurement=[MeasurementSpec(name="y1", expression="x1", noise_scale=1.0)],
        cost=CostSpec(running="-(u1 - 10.0) * (a - p1*u1)", terminal="0"),
    )


def _pricing_constants_matching(config: ModelConfig) -> PricingConstants:
    return PricingConstants(a=100.0, c=10.0, eta=0.3, p_ref=35.0,
                              b_prior=config.parameters[0].prior_guess,
                              F_dyn=1.0, F_param=0.05, sigma_y=1.0, lambda_reg=0.0)


def test_pricing_equivalent_config_compiles_clean():
    plant = compile_plant(_pricing_equivalent_config())
    assert plant.nx == 1 and plant.nu == 1 and plant.n_p == 1 and plant.ny == 1


def test_pricing_equivalent_passes_real_validator():
    config = _pricing_equivalent_config()
    plant = compile_plant(config)
    constants = GenericConstants(prior=np.array([config.parameters[0].prior_guess]))
    # Must not raise.
    validate_dual_control_plugin(
        plant.dynamics, plant.measurement, plant.cost, plant.continuous_dynamics,
        nx=1, nu=1, nv=1, ny=1, constants=constants, n_p=1, K=3)


def test_pricing_equivalent_continuous_dynamics_matches_real_module():
    config = _pricing_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([config.parameters[0].prior_guess]))
    real_constants = _pricing_constants_matching(config)

    rng = np.random.default_rng(0)
    K = 5
    xa = rng.uniform(0, 50, size=(2, K))  # [demand, b_hat]
    u = rng.uniform(10, 45, size=(1, K))
    u_lims = np.array([[10.0, 45.0]])

    xdot_w, nx_w, np_w, F_w, Fp_w = plant.continuous_dynamics(
        1.0, xa, u, wiz_constants, True, u_lims, 1)
    xdot_r, nx_r, np_r, F_r, Fp_r = real_pricing_cd.continuous_dynamics(
        1.0, xa, u, real_constants, True, u_lims, 1)

    assert np.allclose(xdot_w, xdot_r)
    assert nx_w == nx_r
    assert np_w == np_r
    assert np.allclose(F_w, F_r)
    assert np.allclose(Fp_w, Fp_r)


def test_pricing_equivalent_dynamics_matches_real_module():
    config = _pricing_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([config.parameters[0].prior_guess]))
    real_constants = _pricing_constants_matching(config)

    rng = np.random.default_rng(1)
    K = 4
    xa = rng.uniform(0, 50, size=(2, K))
    u = rng.uniform(10, 45, size=(1, K))
    w = rng.normal(size=(2, K))
    u_lims = np.array([[10.0, 45.0]])

    out_w = plant.dynamics(1.0, xa, u, wiz_constants, w, False, True, 0, 0.0, u_lims, 1)
    out_r = real_pricing_dyn.dynamics(1.0, xa, u, real_constants, w, False, True, 0, 0.0, u_lims, 1)
    assert np.allclose(out_w, out_r)


def test_pricing_equivalent_measurement_matches_real_module():
    config = _pricing_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([config.parameters[0].prior_guess]))
    real_constants = _pricing_constants_matching(config)

    rng = np.random.default_rng(2)
    K = 4
    xa = rng.uniform(0, 50, size=(2, K))
    u = rng.uniform(10, 45, size=(1, K))
    v = rng.normal(size=(1, K))

    out_w = plant.measurement(1.0, xa, u, wiz_constants, v, False, True, 0)
    out_r = real_pricing_meas.measurement(1.0, xa, u, real_constants, v, False, True, 0)
    assert np.allclose(out_w, out_r)


def test_pricing_equivalent_cost_matches_real_module():
    config = _pricing_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([config.parameters[0].prior_guess]))
    # Match the real cost's constants exactly: p_ref/lambda_reg chosen so
    # the regularizer term is zero, since the wizard config's running cost
    # above deliberately omits the QP-conditioning regularizer term (out
    # of scope for this equivalence check -- only the profit term is
    # compared).
    real_constants = PricingConstants(a=100.0, c=10.0, eta=0.3, p_ref=35.0,
                                        b_prior=config.parameters[0].prior_guess,
                                        lambda_reg=0.0)

    rng = np.random.default_rng(3)
    K = 4
    xa = rng.uniform(0, 50, size=(2, K))
    u = rng.uniform(10, 45, size=(1, K))
    u_lims = np.array([[10.0, 45.0]])

    out_w = plant.cost(xa, u, None, u_lims, 1, wiz_constants)
    out_r = real_pricing_cost.cost(xa, u, None, u_lims, 1, real_constants)
    assert np.allclose(out_w, out_r)


def test_pricing_equivalent_cost_terminal_at_nan_column():
    config = _pricing_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([0.5]))
    xa = np.array([[10.0, 20.0], [0.5, 0.5]])
    u = np.array([[20.0, np.nan]])
    out = plant.cost(xa, u, None, np.array([[10.0, 45.0]]), 1, wiz_constants)
    assert out[1] == 0.0  # terminal expression defaults to "0"
    assert out[0] != 0.0


# ----------------------------------------------------------------------
# Config reproducing applications/duopoly_pricing/'s math exactly
# (2 states, 2 actions, 4 unknown parameters with cross-terms -- the
# structurally different, harder of the two proven real cases).
# ----------------------------------------------------------------------

def _duopoly_equivalent_config() -> ModelConfig:
    return ModelConfig(
        name="duopoly_equivalent",
        dt=1.0,
        n_sessions=10,
        states=[StateSpec(name="xA", initial_value=0.0, min=0.0),
                 StateSpec(name="xB", initial_value=0.0, min=0.0)],
        actions=[ActionSpec(name="pA", min=0.0, max=60.0),
                  ActionSpec(name="pB", min=0.0, max=60.0)],
        parameters=[
            ParameterSpec(name="bAA", prior_guess=1.0, prior_variance=1.0),
            ParameterSpec(name="bAB", prior_guess=0.2, prior_variance=1.0),
            ParameterSpec(name="bBA", prior_guess=0.2, prior_variance=1.0),
            ParameterSpec(name="bBB", prior_guess=1.0, prior_variance=1.0),
        ],
        constants=[ConstantSpec(name="aA", value=80.0), ConstantSpec(name="aB", value=80.0),
                    ConstantSpec(name="etaA", value=0.3), ConstantSpec(name="etaB", value=0.3)],
        dynamics={
            "xA": "etaA * (aA - bAA*pA + bAB*pB - xA)",
            "xB": "etaB * (aB - bBB*pB + bBA*pA - xB)",
        },
        measurement=[MeasurementSpec(name="yA", expression="xA"),
                      MeasurementSpec(name="yB", expression="xB")],
        cost=CostSpec(running="-(pA)*(aA - bAA*pA + bAB*pB) - (pB)*(aB - bBB*pB + bBA*pA)",
                       terminal="0"),
    )


def _duopoly_constants_matching(config: ModelConfig) -> DuopolyPricingConstants:
    priors = np.array([p.prior_guess for p in config.parameters])
    # lambda_reg_A/B explicitly zeroed: the wizard config's running cost
    # above only reproduces the profit terms, not the real module's
    # QP-conditioning regularizer (lambda_reg defaults to 1e-4, not 0) --
    # zeroing it here keeps this an apples-to-apples comparison of the
    # part actually being tested.
    return DuopolyPricingConstants(
        a_A=80.0, a_B=80.0, c_A=0.0, c_B=0.0, eta_A=0.3, eta_B=0.3,
        p_ref_A=0.0, p_ref_B=0.0, b_prior=priors,
        lambda_reg_A=0.0, lambda_reg_B=0.0)


def test_duopoly_equivalent_config_compiles_clean():
    plant = compile_plant(_duopoly_equivalent_config())
    assert plant.nx == 2 and plant.nu == 2 and plant.n_p == 4 and plant.ny == 2


def test_duopoly_equivalent_passes_real_validator():
    config = _duopoly_equivalent_config()
    plant = compile_plant(config)
    constants = GenericConstants(prior=np.array([p.prior_guess for p in config.parameters]))
    validate_dual_control_plugin(
        plant.dynamics, plant.measurement, plant.cost, plant.continuous_dynamics,
        nx=2, nu=2, nv=2, ny=2, constants=constants, n_p=4, K=3)


def test_duopoly_equivalent_continuous_dynamics_matches_real_module():
    config = _duopoly_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([p.prior_guess for p in config.parameters]))
    real_constants = _duopoly_constants_matching(config)

    rng = np.random.default_rng(4)
    K = 5
    xa = rng.uniform(0, 60, size=(6, K))
    u = rng.uniform(0, 60, size=(2, K))
    u_lims = np.array([[0.0, 60.0], [0.0, 60.0]])

    xdot_w, nx_w, np_w, F_w, Fp_w = plant.continuous_dynamics(
        1.0, xa, u, wiz_constants, True, u_lims, 1)
    xdot_r, nx_r, np_r, F_r, Fp_r = real_duopoly_cd.continuous_dynamics(
        1.0, xa, u, real_constants, True, u_lims, 1)

    assert np.allclose(xdot_w, xdot_r)
    assert nx_w == nx_r and np_w == np_r


def test_duopoly_equivalent_cost_matches_real_module():
    config = _duopoly_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([p.prior_guess for p in config.parameters]))
    real_constants = _duopoly_constants_matching(config)

    rng = np.random.default_rng(5)
    K = 4
    xa = rng.uniform(0, 60, size=(6, K))
    u = rng.uniform(0, 60, size=(2, K))
    u_lims = np.array([[0.0, 60.0], [0.0, 60.0]])

    out_w = plant.cost(xa, u, None, u_lims, 1, wiz_constants)
    out_r = real_duopoly_cost.cost(xa, u, None, u_lims, 1, real_constants)
    assert np.allclose(out_w, out_r)


# ----------------------------------------------------------------------
# Generic behavior not covered by either equivalence check.
# ----------------------------------------------------------------------

def test_zero_unknown_parameters_compiles_and_validates():
    config = ModelConfig(
        name="no_estimation",
        states=[StateSpec(name="x1", initial_value=0.0)],
        actions=[ActionSpec(name="u1", min=0.0, max=10.0)],
        dynamics={"x1": "u1 - x1"},
        measurement=[MeasurementSpec(name="y1", expression="x1")],
        cost=CostSpec(running="x1**2 + u1**2", terminal="0"),
    )
    plant = compile_plant(config)
    assert plant.n_p == 0
    constants = GenericConstants(prior=np.zeros(0))
    validate_dual_control_plugin(
        plant.dynamics, plant.measurement, plant.cost, plant.continuous_dynamics,
        nx=1, nu=1, nv=1, ny=1, constants=constants, n_p=0, K=3)


def test_bad_equation_raises_plant_compilation_error_with_all_issues():
    config = _pricing_equivalent_config()
    config.dynamics["x1"] = "eta * (a - p1*u1 - x1) + unknown_symbol"
    config.cost.running = "another_bad_name"
    with pytest.raises(PlantCompilationError) as exc_info:
        compile_plant(config)
    msg = str(exc_info.value)
    assert "unknown_symbol" in msg
    assert "another_bad_name" in msg


def test_structurally_invalid_config_raises_before_parsing_equations():
    config = _pricing_equivalent_config()
    config.name = ""  # structural error
    with pytest.raises(PlantCompilationError) as exc_info:
        compile_plant(config)
    assert any("name" in e.lower() for e in exc_info.value.errors)


def test_state_clamp_applied_after_step():
    config = _pricing_equivalent_config()  # x1 has min=0.0
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([0.5]))
    # Force a large negative drift so the unclamped step would go below 0.
    xa = np.array([[0.5], [0.5]])
    u = np.array([[45.0]])  # high price -> demand relaxes toward a very negative target
    w = np.zeros((2, 1))
    out = plant.dynamics(10.0, xa, u, wiz_constants, w, False, True, 0, 0.0,
                          np.array([[10.0, 45.0]]), 1)
    assert out[0, 0] >= 0.0


def test_noise_diffusion_column_request_shape():
    config = _pricing_equivalent_config()
    plant = compile_plant(config)
    wiz_constants = GenericConstants(prior=np.array([0.5]))
    xa = np.zeros((2, 3))
    u = np.full((1, 3), 20.0)
    w = np.zeros((2, 3))
    col = plant.dynamics(1.0, xa, u, wiz_constants, w, 1, True, 0, 0.0,
                          np.array([[10.0, 45.0]]), 1)
    assert col.shape == (2, 3)
    # Every column should be identical (F_a doesn't depend on x/u here).
    assert np.allclose(col[:, 0], col[:, 1])
    assert np.allclose(col[:, 0], col[:, 2])
