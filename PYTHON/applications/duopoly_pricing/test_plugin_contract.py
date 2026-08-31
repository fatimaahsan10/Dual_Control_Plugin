"""
Pre-flight contract check for the duopoly-pricing plug-in, using
extensions/dual_control/validate_plugin.py's validate_dual_control_plugin
-- same role as applications/dynamic_pricing/test_plugin_contract.py, now
checking the DynamicsFn/MeasurementFn/CostFn/ContinuousDynamicsFn
Protocols (extensions/dual_control/protocols.py) generalize to nx=2,
nu=2, n_p=4 with BOTH augment_states=True and augment_states=False (the
default modes validate_dual_control_plugin checks), not just the True-only
restriction linear_plant.py needed.

This is a SHAPE/SIGNATURE/NaN-hygiene check on synthetic data, not a
proof the pricing economics are correct -- see validate_plugin.py's own
module docstring for the scope of what it checks.
"""

import numpy as np

from applications.duopoly_pricing.continuous_dynamics import continuous_dynamics
from applications.duopoly_pricing.cost import cost
from applications.duopoly_pricing.dynamics import dynamics
from applications.duopoly_pricing.duopoly_constants import DuopolyPricingConstants
from applications.duopoly_pricing.measurement import measurement
from extensions.dual_control.validate_plugin import validate_dual_control_plugin


def make_constants():
    return DuopolyPricingConstants(
        a_A=100.0, a_B=100.0, c_A=10.0, c_B=12.0, eta_A=0.3, eta_B=0.3,
        p_ref_A=35.0, p_ref_B=35.0, b_prior=np.array([2.0, 0.8, 0.6, 1.8]),
        F_dyn_A=1.0, F_dyn_B=1.0, F_param_A=0.05, F_param_B=0.05,
        sigma_y_A=1.0, sigma_y_B=1.0, lambda_reg_A=1e-4, lambda_reg_B=1e-4)


def test_duopoly_pricing_plugin_satisfies_dual_control_protocols():
    # nx=2 (D_A, D_B realized demand), nu=2 (p_A, p_B), n_p=4 (b_AA, b_AB,
    # b_BA, b_BB), nv=2/ny=2 (both demands observed directly, one noise
    # channel each) -- see continuous_dynamics.py's module docstring for
    # the parameter-ordering convention and the identifiability argument
    # for why both demands (not just "your own") must be observed.
    validate_dual_control_plugin(
        dynamics, measurement, cost, continuous_dynamics,
        nx=2, nu=2, nv=2, ny=2, constants=make_constants(), n_p=4)
