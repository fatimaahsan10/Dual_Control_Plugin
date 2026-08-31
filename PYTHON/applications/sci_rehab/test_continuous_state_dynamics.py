"""
Checks for continuous_state_dynamics.py against ContinuousStateDynamics.m,
at Katie's actual dimensions (nx=6 states, nu=13 WHO therapy doses).

Uses a hand-built SindyModelConstants with poly_order=1 (bias + linear
terms only, no cross/trig) so the resulting Theta*Xi arithmetic can be
verified by hand: with n_vars = nx+nu = 19, the library term order is
[1, x1..x6, u1..u13] (indices 0..19), so term index 7 is u1's linear
coefficient, term index 8 is u2's, etc.
"""

import numpy as np

from applications.sci_rehab.continuous_state_dynamics import continuous_state_dynamics
from applications.sci_rehab.sindy_model_constants import SindyModelConstants

NX = 6
NU = 13
N_TERMS = 1 + (NX + NU)  # bias + linear only (poly_order=1)

U1_TERM = 1 + NX + 0  # term index of u1's linear coefficient
U2_TERM = 1 + NX + 1  # term index of u2's linear coefficient


def make_constants(swat=0.8, xi_idx=None):
    Xi = np.zeros((N_TERMS, NX))
    Xi[U1_TERM, 0] = 0.5   # dSCIM/dt = 0.5 * u1
    Xi[U2_TERM, 1] = 0.3   # dBBS/dt  = 0.3 * u2
    # spurious nonzero coefficients on the STATIC states -- must be
    # zeroed by continuous_state_dynamics regardless of what Xi says
    Xi[U1_TERM, 2] = 99.0
    Xi[U1_TERM, 5] = -42.0

    return SindyModelConstants(
        Xi=Xi,
        lib_cfg={"poly_order": 1, "include_cross": False},
        swat_ceiling=swat,
        xi_idx=np.zeros((0, 2), dtype=int) if xi_idx is None else xi_idx,
        F_dyn=1e-3, F_param=1e-4)


def test_linear_sindy_terms_and_static_states_zeroed():
    constants = make_constants()
    xa = np.array([[0.3, 0.4],
                    [0.3, 0.4],
                    [0.5, 0.5],
                    [0.2, 0.2],
                    [0.1, 0.1],
                    [0.6, 0.6]])
    u = np.zeros((NU, 2))
    u[0, :] = [0.2, 0.4]   # u1
    u[1, :] = [0.1, 0.6]   # u2

    xdot, nx, n_p, F, Fp = continuous_state_dynamics(
        dt=1.0, xa=xa, u=u, constants=constants, augment_states=False,
        u_lims=None, u_lim_method=None)

    assert nx == 6 and n_p == 0
    assert xdot.shape == (6, 2)
    assert np.allclose(xdot[0, :], 0.5 * u[0, :])   # dSCIM/dt
    assert np.allclose(xdot[1, :], 0.3 * u[1, :])   # dBBS/dt
    assert np.allclose(xdot[2:6, :], 0.0)           # static states: always 0
    assert np.allclose(F, 1e-3 * np.eye(6))
    assert Fp.shape == (0, 0)


def test_recovery_ceiling_clips_positive_growth():
    constants = make_constants(swat=0.8)
    # column 0: x1 well below ceiling -> dynamics active
    # column 1: x1 AT the ceiling, u1 > 0 -> raw growth would be
    #           positive -> must be clipped to 0
    xa = np.array([[0.3, 0.8],
                    [0.3, 0.3],
                    [0.5, 0.5], [0.2, 0.2], [0.1, 0.1], [0.6, 0.6]])
    u = np.zeros((NU, 2))
    u[0, :] = [0.5, 0.5]

    xdot, *_ = continuous_state_dynamics(
        dt=1.0, xa=xa, u=u, constants=constants, augment_states=False,
        u_lims=None, u_lim_method=None)

    assert xdot[0, 0] > 0.0     # below ceiling: unclipped
    assert xdot[0, 1] == 0.0    # at ceiling, growing: clipped


def test_recovery_floor_clips_negative_decay():
    constants = make_constants(swat=0.8)
    # column 0: x1 at 0, u1 < 0 -> raw derivative would be negative ->
    # must be clipped to 0 (can't go below the floor)
    xa = np.array([[0.0], [0.3], [0.5], [0.2], [0.1], [0.6]])
    u = np.zeros((NU, 1))
    u[0, :] = [-0.5]

    xdot, *_ = continuous_state_dynamics(
        dt=1.0, xa=xa, u=u, constants=constants, augment_states=False,
        u_lims=None, u_lim_method=None)

    assert xdot[0, 0] == 0.0


def test_augmented_parameter_overwrites_xi_and_matches_xi_idx_rows():
    xi_idx = np.array([[U1_TERM, 0]])  # overwrite u1's SCIM coefficient
    constants = make_constants(xi_idx=xi_idx)

    x_state = np.array([[0.3, 0.3], [0.3, 0.3], [0.5, 0.5],
                          [0.2, 0.2], [0.1, 0.1], [0.6, 0.6]])
    xi_vec = np.array([[2.0, -1.0]])  # per-column overwritten coefficient
    xa = np.vstack([x_state, xi_vec])
    u = np.zeros((NU, 2))
    u[0, :] = [0.1, 0.1]

    xdot, nx, n_p, F, Fp = continuous_state_dynamics(
        dt=1.0, xa=xa, u=u, constants=constants, augment_states=True,
        u_lims=None, u_lim_method=None)

    assert n_p == 1
    assert xdot.shape == (7, 2)  # 6 states + 1 augmented parameter
    # overwritten coefficient (2.0, then -1.0), NOT the original 0.5
    assert np.isclose(xdot[0, 0], 2.0 * 0.1)
    assert np.isclose(xdot[0, 1], -1.0 * 0.1)
    assert np.allclose(xdot[6, :], 0.0)  # parameter dynamics: zero-mean random walk
    assert np.allclose(Fp, 1e-4 * np.eye(1))


def test_np_mismatch_raises():
    xi_idx = np.array([[U1_TERM, 0], [U2_TERM, 1]])  # expects 2 params
    constants = make_constants(xi_idx=xi_idx)

    xa = np.vstack([np.full((6, 1), 0.3), np.array([[1.0]])])  # only 1 param
    u = np.zeros((NU, 1))

    try:
        continuous_state_dynamics(
            dt=1.0, xa=xa, u=u, constants=constants, augment_states=True,
            u_lims=None, u_lim_method=None)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_control_squashing_applied_before_library_evaluation():
    constants = make_constants()
    xa = np.array([[0.3], [0.3], [0.5], [0.2], [0.1], [0.6]])
    u_raw = np.zeros((NU, 1))
    u_raw[0, 0] = 10.0  # far outside [-1, 1]; tanh should saturate it near 1
    u_lims = np.tile(np.array([[-1.0, 1.0]]), (NU, 1))

    xdot, *_ = continuous_state_dynamics(
        dt=1.0, xa=xa, u=u_raw, constants=constants, augment_states=False,
        u_lims=u_lims, u_lim_method=2)

    u1_squashed = np.tanh(10.0)  # lims are [-1,1] -> squash formula reduces to tanh
    assert np.isclose(xdot[0, 0], 0.5 * u1_squashed)


if __name__ == "__main__":
    tests = [test_linear_sindy_terms_and_static_states_zeroed,
              test_recovery_ceiling_clips_positive_growth,
              test_recovery_floor_clips_negative_decay,
              test_augmented_parameter_overwrites_xi_and_matches_xi_idx_rows,
              test_np_mismatch_raises,
              test_control_squashing_applied_before_library_evaluation]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
