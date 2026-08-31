"""
Pre-flight contract check for the dynamic-pricing plug-in, using
extensions/dual_control/validate_plugin.py's validate_dual_control_plugin
-- the real second, independent test of whether the DynamicsFn/
MeasurementFn/CostFn/ContinuousDynamicsFn Protocols formally documented
in extensions/dual_control/protocols.py genuinely generalize beyond
Katie's SCI plant (the plant the Protocols were originally transcribed
from).

This is a SHAPE/SIGNATURE/NaN-hygiene check on synthetic data, not a
proof the pricing economics are correct -- see validate_plugin.py's own
module docstring for the scope of what it checks.
"""

from applications.dynamic_pricing.continuous_dynamics import continuous_dynamics
from applications.dynamic_pricing.cost import cost
from applications.dynamic_pricing.dynamics import dynamics
from applications.dynamic_pricing.measurement import measurement
from applications.dynamic_pricing.pricing_constants import PricingConstants
from extensions.dual_control.validate_plugin import validate_dual_control_plugin


def make_constants():
    return PricingConstants(
        a=100.0, c=10.0, eta=0.3, p_ref=35.0, b_prior=2.0,
        F_dyn=1.0, F_param=0.05, sigma_y=1.0, lambda_reg=1e-4)


def test_dynamic_pricing_plugin_satisfies_dual_control_protocols():
    # nx=1 (realized demand), nu=1 (price), n_p=1 (price sensitivity b),
    # nv=1/ny=1 (demand observed directly, one noise channel) -- see
    # CLAUDE.md's dynamic-pricing math review for the full derivation.
    validate_dual_control_plugin(
        dynamics, measurement, cost, continuous_dynamics,
        nx=1, nu=1, nv=1, ny=1, constants=make_constants(), n_p=1)
