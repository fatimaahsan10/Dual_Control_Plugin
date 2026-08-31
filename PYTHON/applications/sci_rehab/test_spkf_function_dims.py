"""
Confirms spkf_function.py -- ported from Andrew's
CODES/ANDREW'S CODE/SPKF_function.m -- works unchanged at Katie's state
dimension (6 states), same approach as the other test_*_dims.py files
in this directory.

Diff check: Katie's SPKF_function.m is 466 lines vs Andrew's 165, but
`Mode = 0` is hardcoded at line 12 in BOTH files (confirmed by reading
both headers directly) and the extra ~300 lines are entirely inside a
dead `elseif Mode == 1` branch (a nested-CSP variant) that can never
execute. Diffing just the live `Mode == 0` path against Andrew's file
(Compare-Object) shows it is reformatted (wrapped in an explicit
`if Mode == 0` block) but not logically changed, and the function
signature `[x_hat_all, P_x] = SPKF_function(dt, Dynamics_F,
Measurement_F, z_true_all, x_hat, w_hat, v_hat, P_x, P_w, P_v, h,
AdditiveNoise)` is byte-identical between the two files. So this is a
reuse-as-is case, same conclusion CLAUDE.md already draws for this file.

spkf_function.py never hardcodes a state/measurement dimension --
nx/L/nz are all read off x_hat/xa_hat/Z shapes -- so this file imports
it directly (no duplication) and re-runs
andrew/test_spkf_function.py's "matches the analytic linear Kalman
filter" check at nx=nz=6 instead of nx=nz=3.
"""

import numpy as np

from extensions.dual_control.spkf_function import spkf_function

# Katie's state dimension (SCIM, BBS, AIS, Age, DPI, Caregiver support).
N_STATES = 6


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


def test_spkf_matches_linear_kalman_filter_at_6_states():
    rng = np.random.default_rng(0)
    nx = nz = N_STATES
    iterations = 6

    A = 0.9 * np.eye(nx) + 0.03 * rng.standard_normal((nx, nx))
    H = np.eye(nz, nx)
    Qm = rng.standard_normal((nx, nx))
    Q = 0.05 * (Qm @ Qm.T + np.eye(nx))
    Rm = rng.standard_normal((nz, nz))
    R = 0.05 * (Rm @ Rm.T + np.eye(nz))

    def dynamics_f(dt, X_x, X_w):
        return A @ X_x + X_w

    def measurement_f(X_x, X_v):
        return H @ X_x + X_v

    x0 = rng.standard_normal(nx)
    P0 = 0.1 * np.eye(nx)
    z_all = rng.standard_normal((iterations, nz))

    x_ref_all, P_ref = linear_kf_reference(A, H, Q, R, x0, P0, z_all)

    x_hat_all, P_x = spkf_function(
        dt=1.0, dynamics_f=dynamics_f, measurement_f=measurement_f,
        z_true_all=z_all, x_hat=x0.copy(), w_hat=np.zeros(nx),
        v_hat=np.zeros(nz), P_x=P0.copy(), P_w=Q, P_v=R, h=np.sqrt(3),
        additive_noise=False)

    assert x_hat_all.shape == (nx, iterations)
    assert np.allclose(x_hat_all, x_ref_all, atol=1e-8), (
        "SPKF state estimates diverge from the analytic linear KF at n=6"
    )
    assert np.allclose(P_x, P_ref, atol=1e-8), (
        "SPKF final covariance diverges from the analytic linear KF at n=6"
    )


if __name__ == "__main__":
    tests = [test_spkf_matches_linear_kalman_filter_at_6_states]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
