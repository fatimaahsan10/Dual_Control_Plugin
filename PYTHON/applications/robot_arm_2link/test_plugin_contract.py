"""
Pre-flight contract check for the robot-arm plug-in, using
extensions/dual_control/validate_plugin.py's validate_dual_control_plugin
-- same role as applications/duopoly_pricing/test_plugin_contract.py, now
checking the DynamicsFn/MeasurementFn/CostFn/ContinuousDynamicsFn
Protocols (extensions/dual_control/protocols.py) generalize to nx=4,
nu=2, n_p=1 with BOTH augment_states=True and augment_states=False (the
default modes validate_dual_control_plugin checks).

This is a SHAPE/SIGNATURE/NaN-hygiene check on synthetic data, not a
proof the arm's physics are correct -- see validate_plugin.py's own
module docstring for the scope of what it checks. In particular, the
synthetic states validate_dynamics_fn/validate_cost_fn draw are uniform
in [0, 1] for EVERY row, including m2 -- continuous_dynamics.py's
docstring already confirms M(q) stays invertible even at m2 -> 0 (as
long as I2 > 0), which is exactly why this passes without needing any
special-casing for the near-zero-mass corner this synthetic sampling can
hit.
"""

import numpy as np

from applications.robot_arm_2link.arm_constants import RobotArmConstants
from applications.robot_arm_2link.continuous_dynamics import continuous_dynamics
from applications.robot_arm_2link.cost import cost
from applications.robot_arm_2link.dynamics import dynamics
from applications.robot_arm_2link.measurement import measurement
from extensions.dual_control.validate_plugin import validate_dual_control_plugin


def make_constants():
    return RobotArmConstants(
        l1=1.0, l2=1.0, lc1=0.5, lc2=0.5, m1=1.0, I1=0.1, I2=0.1, g=9.81,
        fv1=0.5, fv2=0.5, m2_prior=0.5, q1_target=np.pi / 3, q2_target=-np.pi / 4)


def test_robot_arm_plugin_satisfies_dual_control_protocols():
    # nx=4 (q1, q2, w1, w2), nu=2 (tau1, tau2), n_p=1 (m2), nv=4/ny=4
    # (all four physical states observed directly, one noise channel
    # each) -- see continuous_dynamics.py/measurement.py's module
    # docstrings.
    validate_dual_control_plugin(
        dynamics, measurement, cost, continuous_dynamics,
        nx=4, nu=2, nv=4, ny=4, constants=make_constants(), n_p=1)
