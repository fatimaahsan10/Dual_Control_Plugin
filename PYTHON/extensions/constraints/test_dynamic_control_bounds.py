"""
Tests for dynamic_control_bounds.py.

Primary verification case: the paper's own inverted-pendulum example
(Sec IV-A, eq 20-23). State x=[phi, omega], constraint omega < 1, i.e.
h(x) = 1 - omega >= 0. Relative degree one w.r.t. u (hand-derived in the
module's companion explanation, not yet via relative_degree_reduction.py
which doesn't exist yet):

    continuous dynamics: phi_dot = omega, omega_dot = sin(phi) + u
    dh/dx = [0, -1]
    h_dot(x) = dh/dx . f(x,u) = -(sin(phi) + u)
    h_tilde(x,u) = h_dot(x) + alpha*h(x) = alpha*(1-omega) - sin(phi) - u   (eq 19)

Since h_tilde is exactly affine in u (dh_tilde/du = -1, constant), the
module's first-order approximation is exact here, not just approximate --
a strong test: any tolerance failure indicates a real bug, not a
linearization artifact.
"""

import numpy as np
import pytest

from extensions.constraints.dynamic_control_bounds import (
    build_time_varying_lims,
    solve_box_from_constraint,
)

ALPHA = 0.1


def h_tilde_pendulum(x, u):
    phi, omega = x[0], x[1]
    return np.array([ALPHA * (1.0 - omega) - np.sin(phi) - u[0]])


def dh_tilde_du_pendulum(x, u):
    return np.array([[-1.0]])


def closed_form_u_max(phi, omega):
    return ALPHA * (1.0 - omega) - np.sin(phi)


@pytest.mark.parametrize("phi,omega,u_i", [
    (-np.pi, 0.0, 0.0),
    (np.pi / 4, 0.5, 0.2),
    (0.0, -0.8, -1.5),
    (2.5, 0.99, 3.0),
])
def test_pendulum_upper_bound_matches_closed_form_analytic_jacobian(phi, omega, u_i):
    x_i = np.array([phi, omega])
    lo, hi = solve_box_from_constraint(
        h_tilde_pendulum, x_i, np.array([u_i]), dhdu_fn=dh_tilde_du_pendulum)
    assert lo[0] == -np.inf
    assert hi[0] == pytest.approx(closed_form_u_max(phi, omega), abs=1e-10)


@pytest.mark.parametrize("phi,omega,u_i", [
    (-np.pi, 0.0, 0.0),
    (np.pi / 4, 0.5, 0.2),
    (0.0, -0.8, -1.5),
])
def test_pendulum_upper_bound_matches_closed_form_finite_difference_jacobian(phi, omega, u_i):
    # dhdu_fn=None -> numeric Jacobian path (reuses andrew/finite_difference.py).
    # h_tilde is exactly affine in u, so forward differencing has zero
    # truncation error here -- this should match to near machine precision,
    # not just "close enough for a linearization".
    x_i = np.array([phi, omega])
    lo, hi = solve_box_from_constraint(h_tilde_pendulum, x_i, np.array([u_i]))
    assert lo[0] == -np.inf
    assert hi[0] == pytest.approx(closed_form_u_max(phi, omega), abs=1e-6)


def test_intersects_with_preexisting_hard_limits():
    x_i = np.array([np.pi / 4, 0.5])  # closed-form u_max ~= -0.657
    lo, hi = solve_box_from_constraint(
        h_tilde_pendulum, x_i, np.array([0.2]), dhdu_fn=dh_tilde_du_pendulum,
        lower=-5.0, upper=5.0)
    assert lo[0] == -5.0  # hard lower survives -- constraint only pins upper
    assert hi[0] == pytest.approx(closed_form_u_max(np.pi / 4, 0.5), abs=1e-10)

    # hard limit tighter than the constraint-derived bound should win instead
    lo2, hi2 = solve_box_from_constraint(
        h_tilde_pendulum, x_i, np.array([0.2]), dhdu_fn=dh_tilde_du_pendulum,
        lower=-5.0, upper=-1.0)
    assert hi2[0] == -1.0


def test_build_time_varying_lims_over_a_trajectory():
    x_traj = np.array([
        [-np.pi, -np.pi / 2, 0.0],
        [0.0, 0.3, 0.6],
    ])
    u_traj = np.zeros((1, 3))
    lims = build_time_varying_lims(
        h_tilde_pendulum, x_traj, u_traj, dhdu_fn=dh_tilde_du_pendulum)

    assert lims.shape == (3, 1, 2)
    for i in range(3):
        phi, omega = x_traj[:, i]
        assert lims[i, 0, 0] == -np.inf
        assert lims[i, 0, 1] == pytest.approx(closed_form_u_max(phi, omega), abs=1e-10)


def test_row_depending_on_two_control_channels_raises():
    def h_two_channel(x, u):
        return np.array([u[0] + u[1] - 1.0])

    with pytest.raises(ValueError, match="not expressible as an axis-aligned box"):
        solve_box_from_constraint(h_two_channel, np.zeros(1), np.array([0.1, 0.2]))


def test_row_independent_of_control_warns_and_leaves_bound_untouched():
    def h_state_only(x, u):
        return np.array([x[0]])  # no dependence on u at all

    with pytest.warns(RuntimeWarning, match="relative degree is not one"):
        lo, hi = solve_box_from_constraint(
            h_state_only, np.array([5.0]), np.array([0.0]), lower=-2.0, upper=2.0)
    assert lo[0] == -2.0
    assert hi[0] == 2.0


def test_vector_h_two_independent_scalar_rows_pin_different_bounds():
    # row 0: u >= 0.3 ; row 1: u <= 0.7  -- both relative degree 1, single
    # channel each, so this SHOULD reduce to a clean box.
    def h_vec(x, u):
        return np.array([u[0] - 0.3, 0.7 - u[0]])

    def dhdu_vec(x, u):
        return np.array([[1.0], [-1.0]])

    lo, hi = solve_box_from_constraint(
        h_vec, np.zeros(1), np.array([0.5]), dhdu_fn=dhdu_vec)
    assert lo[0] == pytest.approx(0.3)
    assert hi[0] == pytest.approx(0.7)
