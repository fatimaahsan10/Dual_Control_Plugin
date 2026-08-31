"""
Checks for discrete_state_dynamics.py against DiscreteStateDynamics.m,
at Katie's actual dimensions (nx=6 states, nu=13 WHO therapy doses).

Reuses the same hand-verifiable linear-SINDy setup as
test_continuous_state_dynamics.py (poly_order=1, only u1->dSCIM and
u2->dBBS nonzero) so the Euler step and noise terms can be checked by
hand on top of an already-verified continuous_state_dynamics.py.
"""

import numpy as np

from applications.sci_rehab.discrete_state_dynamics import discrete_state_dynamics
from applications.sci_rehab.sindy_model_constants import SindyModelConstants

NX = 6
NU = 13
N_TERMS = 1 + (NX + NU)  # bias + linear only (poly_order=1)
U1_TERM = 1 + NX + 0
U2_TERM = 1 + NX + 1


def make_constants(swat=0.8, xi_idx=None, F_dyn=1e-3, F_param=1e-4):
    Xi = np.zeros((N_TERMS, NX))
    Xi[U1_TERM, 0] = 0.5   # dSCIM/dt = 0.5 * u1
    Xi[U2_TERM, 1] = 0.3   # dBBS/dt  = 0.3 * u2
    return SindyModelConstants(
        Xi=Xi, lib_cfg={"poly_order": 1, "include_cross": False},
        swat_ceiling=swat,
        xi_idx=np.zeros((0, 2), dtype=int) if xi_idx is None else xi_idx,
        F_dyn=F_dyn, F_param=F_param)


def make_xa(n_cols):
    xa = np.zeros((NX, n_cols))
    xa[0, :] = 0.3
    xa[1, :] = 0.3
    xa[2:, :] = np.array([[0.5], [0.2], [0.1], [0.6]])
    return xa


def test_deterministic_euler_step_matches_hand_computation():
    constants = make_constants()
    xa = make_xa(2)
    u = np.zeros((NU, 2))
    u[0, :] = [0.2, 0.4]   # u1
    u[1, :] = [0.1, 0.6]   # u2
    w = np.zeros((NX, 2))  # nw = nx + n_p = 6 (not augmented)
    dt = 2.0

    xkplus1 = discrete_state_dynamics(
        dt=dt, xa=xa, u=u, constants=constants, w=w, noise_index=False,
        augment_states=False, w_from_filter=0, lam=0.0, u_lims=None,
        u_lim_method=None)

    expected_f0 = 0.5 * u[0, :]
    expected_f1 = 0.3 * u[1, :]
    assert np.allclose(xkplus1[0, :], xa[0, :] + dt * expected_f0)
    assert np.allclose(xkplus1[1, :], xa[1, :] + dt * expected_f1)
    assert np.allclose(xkplus1[2:, :], xa[2:, :])  # static states: unchanged


def test_noise_scaled_by_diffusion_matrix_and_sqrt_dt():
    constants = make_constants(F_dyn=0.02, F_param=0.01)
    xa = make_xa(1)
    u = np.zeros((NU, 1))
    w = np.array([[1.0], [2.0], [0.0], [0.0], [0.0], [0.0]])
    dt = 4.0  # sqrt(dt) = 2, easy to check by hand

    xkplus1 = discrete_state_dynamics(
        dt=dt, xa=xa, u=u, constants=constants, w=w, noise_index=False,
        augment_states=False, w_from_filter=0, lam=0.0, u_lims=None,
        u_lim_method=None)

    # F_a = F_dyn * I(6) here (not augmented) -> noise = F_dyn*w*sqrt(dt)
    expected_noise = 0.02 * w * np.sqrt(dt)
    fa_expected = np.zeros((NX, 1))  # u=0 -> f=0
    assert np.allclose(xkplus1, xa + dt * fa_expected + expected_noise)


def test_w_from_filter_uses_w_directly_without_scaling():
    constants = make_constants()
    xa = make_xa(1)
    u = np.zeros((NU, 1))
    w = np.full((NX, 1), 0.05)

    xkplus1 = discrete_state_dynamics(
        dt=1.0, xa=xa, u=u, constants=constants, w=w, noise_index=False,
        augment_states=False, w_from_filter=1, lam=0.0, u_lims=None,
        u_lim_method=None)

    assert np.allclose(xkplus1, xa + w)  # f=0 (u=0), noise = w unscaled


def test_noise_index_returns_diffusion_column_broadcast():
    constants = make_constants(F_dyn=0.02, F_param=0.01)
    xa = make_xa(3)
    u = np.zeros((NU, 3))

    col1 = discrete_state_dynamics(
        dt=1.0, xa=xa, u=u, constants=constants, w=np.zeros((NX, 3)),
        noise_index=1, augment_states=False, w_from_filter=0, lam=0.0,
        u_lims=None, u_lim_method=None)

    expected = np.zeros((NX, 1))
    expected[0, 0] = 0.02  # F_a = F_dyn*I(6); column 1 is [F_dyn,0,0,0,0,0]
    assert np.allclose(col1, np.tile(expected, (1, 3)))


def test_broadcasts_single_column_w_and_u_across_batch():
    constants = make_constants()
    xa = make_xa(3)
    u_single = np.zeros((NU, 1))
    u_single[0, 0] = 0.2
    w_single = np.zeros((NX, 1))

    xkplus1 = discrete_state_dynamics(
        dt=1.0, xa=xa, u=u_single, constants=constants, w=w_single,
        noise_index=False, augment_states=False, w_from_filter=0, lam=0.0,
        u_lims=None, u_lim_method=None)

    assert xkplus1.shape == (NX, 3)
    assert np.allclose(xkplus1[0, :], xa[0, :] + 0.5 * 0.2)


def test_augmented_case_includes_parameter_block_in_diffusion_matrix():
    xi_idx = np.array([[U1_TERM, 0]])
    constants = make_constants(xi_idx=xi_idx, F_dyn=0.02, F_param=0.01)

    x_state = make_xa(1)
    xi_vec = np.array([[1.0]])
    xa = np.vstack([x_state, xi_vec])
    u = np.zeros((NU, 1))
    w = np.zeros((NX + 1, 1))
    w[6, 0] = 1.0  # noise on the augmented parameter channel only
    dt = 4.0

    xkplus1 = discrete_state_dynamics(
        dt=dt, xa=xa, u=u, constants=constants, w=w, noise_index=False,
        augment_states=True, w_from_filter=0, lam=0.0, u_lims=None,
        u_lim_method=None)

    assert xkplus1.shape == (NX + 1, 1)
    # parameter dynamics are 0 (random walk), so xkplus1[6] = xi_vec + F_param*w[6]*sqrt(dt)
    assert np.isclose(xkplus1[6, 0], 1.0 + 0.01 * 1.0 * np.sqrt(dt))


if __name__ == "__main__":
    tests = [test_deterministic_euler_step_matches_hand_computation,
              test_noise_scaled_by_diffusion_matrix_and_sqrt_dt,
              test_w_from_filter_uses_w_directly_without_scaling,
              test_noise_index_returns_diffusion_column_broadcast,
              test_broadcasts_single_column_w_and_u_across_batch,
              test_augmented_case_includes_parameter_block_in_diffusion_matrix]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
