"""
Tests for relative_degree_reduction.py.

Primary verification case: the paper's own inverted-pendulum example
(Sec IV-A, eq 20-23). Constraint omega < 1 -> h(x) = 1 - omega. Continuous
dynamics implied by the paper's Euler-discretized eq (20):
    phi_dot   = omega
    omega_dot = sin(phi) + u

h(x) = 1 - omega is LINEAR in x, so dh/dx = [0, -1] exactly regardless of
finite-difference step size (forward differencing a linear function has
zero truncation error) -- this makes both the analytic-Jacobian and
numeric-Jacobian paths exact tests, not just "close enough".

Hand-derived closed form (eq 19, alpha=0.1):
    h_tilde(x,u) = 0.1*(1-omega) - sin(phi) - u

This is the SAME closed form test_dynamic_control_bounds.py's
h_tilde_pendulum was hand-written to match -- here it's produced
automatically by the reduction instead.
"""

import numpy as np
import pytest

from extensions.constraints.dynamic_control_bounds import solve_box_from_constraint
from extensions.constraints.relative_degree_reduction import (
    lie_derivative_h,
    state_constraint_to_control_constraint,
)

ALPHA = 0.1


def h_omega(x):
    return np.array([1.0 - x[1]])


def dh_omega_dx(x):
    return np.array([[0.0, -1.0]])


def pendulum_dynamics(x, u):
    phi, omega = x[0], x[1]
    return np.array([omega, np.sin(phi) + u[0]])


def closed_form_h_dot(phi, omega, u):
    return -(np.sin(phi) + u)


def closed_form_h_tilde(phi, omega, u):
    return ALPHA * (1.0 - omega) - np.sin(phi) - u


def closed_form_u_max(phi, omega):
    return ALPHA * (1.0 - omega) - np.sin(phi)


@pytest.mark.parametrize("phi,omega,u", [
    (-np.pi, 0.0, 0.0),
    (np.pi / 4, 0.5, 0.2),
    (0.0, -0.8, -1.5),
    (2.5, 0.99, 3.0),
])
class TestLieDerivative:
    def test_analytic_jacobian(self, phi, omega, u):
        x = np.array([phi, omega])
        h_dot = lie_derivative_h(h_omega, pendulum_dynamics, x, np.array([u]),
                                   dhdx_fn=dh_omega_dx)
        assert h_dot[0] == pytest.approx(closed_form_h_dot(phi, omega, u), abs=1e-10)

    def test_finite_difference_jacobian(self, phi, omega, u):
        # h is affine in x -> forward differencing is exact here, so this
        # should match to near machine precision, not just "linearization
        # accurate".
        x = np.array([phi, omega])
        h_dot = lie_derivative_h(h_omega, pendulum_dynamics, x, np.array([u]))
        assert h_dot[0] == pytest.approx(closed_form_h_dot(phi, omega, u), abs=1e-6)


@pytest.mark.parametrize("phi,omega,u", [
    (-np.pi, 0.0, 0.0),
    (np.pi / 4, 0.5, 0.2),
    (0.0, -0.8, -1.5),
])
def test_state_constraint_to_control_constraint_matches_closed_form(phi, omega, u):
    h_tilde_fn = state_constraint_to_control_constraint(
        h_omega, pendulum_dynamics, ALPHA, dhdx_fn=dh_omega_dx)
    x = np.array([phi, omega])
    val = h_tilde_fn(x, np.array([u]))
    assert val[0] == pytest.approx(closed_form_h_tilde(phi, omega, u), abs=1e-10)


def test_state_constraint_to_control_constraint_numeric_jacobian_matches_too():
    h_tilde_fn = state_constraint_to_control_constraint(
        h_omega, pendulum_dynamics, ALPHA)  # dhdx_fn=None -> numeric path
    x = np.array([np.pi / 4, 0.5])
    val = h_tilde_fn(x, np.array([0.2]))
    assert val[0] == pytest.approx(closed_form_h_tilde(np.pi / 4, 0.5, 0.2), abs=1e-6)


def test_alpha_must_be_positive():
    with pytest.raises(ValueError, match="alpha must be > 0"):
        state_constraint_to_control_constraint(h_omega, pendulum_dynamics, 0.0)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        state_constraint_to_control_constraint(h_omega, pendulum_dynamics, -0.1)


class TestEndToEndCompositionWithDynamicControlBounds:
    """
    The real point of Variant B: h_tilde_fn generated here should be a
    drop-in h_fn for dynamic_control_bounds.solve_box_from_constraint,
    and the resulting bound should match the SAME closed-form u_max
    test_dynamic_control_bounds.py verified independently (there, against
    a hand-derived h_tilde; here, against one built automatically by the
    reduction from nothing but h(x)=1-omega and the raw dynamics).
    """

    @pytest.mark.parametrize("phi,omega", [
        (-np.pi, 0.0),
        (np.pi / 4, 0.5),
        (2.5, 0.99),
    ])
    def test_reduced_constraint_reproduces_pendulum_bound(self, phi, omega):
        h_tilde_fn = state_constraint_to_control_constraint(
            h_omega, pendulum_dynamics, ALPHA, dhdx_fn=dh_omega_dx)
        x_i = np.array([phi, omega])

        lo, hi = solve_box_from_constraint(h_tilde_fn, x_i, np.array([0.0]))

        assert lo[0] == -np.inf
        assert hi[0] == pytest.approx(closed_form_u_max(phi, omega), abs=1e-6)

    def test_fully_numeric_pipeline_both_jacobians_from_finite_differences(self):
        # No analytic dh/dx and no analytic dh_tilde/du anywhere -- both
        # reduction and bound-solving fall back to finite differences.
        h_tilde_fn = state_constraint_to_control_constraint(
            h_omega, pendulum_dynamics, ALPHA)  # numeric dh/dx
        x_i = np.array([0.3, -0.2])

        lo, hi = solve_box_from_constraint(h_tilde_fn, x_i, np.array([0.0]))

        assert lo[0] == -np.inf
        assert hi[0] == pytest.approx(closed_form_u_max(0.3, -0.2), abs=1e-4)


def test_relative_degree_zero_constraint_is_independent_of_u():
    # A constraint whose Lie derivative genuinely doesn't touch u (relative
    # degree > 1 here) -- h(x) = x[0] - 5, dynamics' first slot doesn't
    # depend on u at all.
    def h_x0(x):
        return np.array([x[0] - 5.0])

    def dhdx_x0(x):
        return np.array([[1.0, 0.0]])

    def dynamics_no_u_in_slot0(x, u):
        return np.array([np.cos(x[1]), u[0]])  # x_dot[0] has no u

    h_tilde_fn = state_constraint_to_control_constraint(
        h_x0, dynamics_no_u_in_slot0, alpha=0.5, dhdx_fn=dhdx_x0)
    x = np.array([1.0, 2.0])
    val_u0 = h_tilde_fn(x, np.array([0.0]))
    val_u5 = h_tilde_fn(x, np.array([5.0]))
    assert val_u0[0] == pytest.approx(val_u5[0])  # truly independent of u

    with pytest.warns(RuntimeWarning, match="relative degree is not one"):
        solve_box_from_constraint(h_tilde_fn, x, np.array([0.0]))


def test_vector_h_two_independent_state_constraints():
    def h_vec(x):
        return np.array([1.0 - x[1], x[0] + 3.0])

    def dh_vec_dx(x):
        return np.array([[0.0, -1.0], [1.0, 0.0]])

    def dyn(x, u):
        return np.array([x[1], np.sin(x[0]) + u[0]])

    h_tilde_fn = state_constraint_to_control_constraint(
        h_vec, dyn, ALPHA, dhdx_fn=dh_vec_dx)
    x = np.array([0.2, 0.4])
    val = h_tilde_fn(x, np.array([0.1]))
    assert val.shape == (2,)
    # row 0 depends on u (omega constraint); row 1 doesn't (position-only,
    # relative degree > 1 through this dynamics) -- confirmed independently
    assert val[0] == pytest.approx(closed_form_h_tilde(0.2, 0.4, 0.1), abs=1e-10)
    assert val[1] == pytest.approx(ALPHA * (0.2 + 3.0) + 0.4, abs=1e-10)
