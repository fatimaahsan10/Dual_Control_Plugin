"""
Checks for spkf_function.py.

For a genuinely linear dynamics/measurement model, sigma-point filters
reproduce the exact analytic Kalman filter mean/covariance (the sigma
points capture the true covariance exactly, and a linear map of them
reconstructs the exact transformed covariance) -- so cross-checking
against a hand-written linear KF, run on the same measurement sequence,
is a strong independent verification, not just an approximation check.
"""

import numpy as np

from extensions.dual_control.spkf_function import spkf_function


def linear_kf_reference(A, H, Q, R, x0, P0, z_all):
    x = x0.copy()
    P = P0.copy()
    x_all = np.zeros((x0.shape[0], z_all.shape[0]))
    for k in range(z_all.shape[0]):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        S = H @ P_pred @ H.T + R
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
        x = x_pred + K @ (z_all[k, :] - H @ x_pred)
        P = P_pred - K @ S @ K.T
        x_all[:, k] = x
    return x_all, P


def test_matches_linear_kalman_filter():
    rng = np.random.default_rng(0)
    nx, nz, iterations = 3, 3, 6

    A = 0.9 * np.eye(nx) + 0.03 * rng.standard_normal((nx, nx))
    H = np.eye(nz, nx)
    Qm = rng.standard_normal((nx, nx))
    Q = 0.05 * (Qm @ Qm.T + np.eye(nx))
    Rm = rng.standard_normal((nz, nz))
    R = 0.05 * (Rm @ Rm.T + np.eye(nz))

    def dynamics_f(dt, X_x, X_w):
        return A @ X_x + X_w  # additive noise, via augmented sigma points

    def measurement_f(X_x, X_v):
        return H @ X_x + X_v

    x0 = rng.standard_normal(nx)
    P0 = 0.1 * np.eye(nx)
    z_all = rng.standard_normal((iterations, nz))

    x_ref_all, P_ref = linear_kf_reference(A, H, Q, R, x0, P0, z_all)

    x_hat_all, P_x = spkf_function(
        dt=0.1, dynamics_f=dynamics_f, measurement_f=measurement_f,
        z_true_all=z_all, x_hat=x0.copy(), w_hat=np.zeros(nx),
        v_hat=np.zeros(nz), P_x=P0.copy(), P_w=Q, P_v=R, h=np.sqrt(3),
        additive_noise=False)

    assert x_hat_all.shape == (nx, iterations)
    assert np.allclose(x_hat_all, x_ref_all, atol=1e-8), (
        "SPKF state estimates diverge from the analytic linear KF"
    )
    assert np.allclose(P_x, P_ref, atol=1e-8), (
        "SPKF final covariance diverges from the analytic linear KF"
    )


def test_len_w_hat_mismatch_raises():
    nx = 3
    try:
        spkf_function(
            dt=0.1, dynamics_f=lambda dt, x, w: x, measurement_f=lambda x, v: x,
            z_true_all=np.zeros((1, nx)), x_hat=np.zeros(nx),
            w_hat=np.zeros(nx - 1),  # deliberately mismatched
            v_hat=np.zeros(nx), P_x=np.eye(nx), P_w=np.eye(nx - 1),
            P_v=np.eye(nx), h=np.sqrt(3), additive_noise=False)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [test_matches_linear_kalman_filter, test_len_w_hat_mismatch_raises]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
