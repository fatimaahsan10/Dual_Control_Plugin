"""
Sigma-point (Central Difference) Kalman Filter.

Ported from Andrew's CODES/ANDREW'S CODE/SPKF_function.m.

Represents the belief as 2L+1 deterministic sigma points around the mean
(L = augmented state dimension), pushes each through the real nonlinear
dynamics/measurement functions, and reconstructs mean/covariance from the
transformed points -- no analytic Jacobians needed, unlike an EKF.

Callable contract (already-closed-over callables, matching how
Main_Outer_Control_Loop.m builds Dynamics_F/Measurement_F as closures over
the current control/constants):

    dynamics_f(dt, X_x, W)   -> (nx, N_points) propagated state sigma points
    measurement_f(X_x, V)    -> (nz, N_points) predicted measurement points

Simplification relative to the literal MATLAB source: P_x_all, Xs_before,
x_hat_before, Px_before, Xs_after are computed every iteration in
SPKF_function.m but never appear in its return signature ([x_hat_all, P_x]
only) -- dropped here, same reasoning as the dead code found and dropped
in ilqg_function.py's line search.

DEVIATION FROM THE LITERAL MATLAB SOURCE: x_hat_all is returned as
(nx, iterations) here, not (iterations, nx) -- consistent with the (dim, N)
convention used by every other module in this port (forward_pass.py,
backward_pass.py, ...), rather than this function's own reversed
convention (a byproduct of its `x_hat_all(k,:) = x_hat` row-assignment
style, not a deliberate design choice elsewhere in the codebase).

NOTE: the non-additive branch requires len(w_hat) == len(x_hat) -- see
module-level discussion; this port makes that an explicit checked
precondition instead of a silent indexing assumption.
"""

import numpy as np

from extensions.dual_control.make_pd import make_pd


def _block_diag(*mats):
    mats = [np.atleast_2d(m) for m in mats]
    total = sum(m.shape[0] for m in mats)
    out = np.zeros((total, total))
    i = 0
    for m in mats:
        n = m.shape[0]
        out[i:i + n, i:i + n] = m
        i += n
    return out


def spkf_function(dt, dynamics_f, measurement_f, z_true_all, x_hat, w_hat,
                    v_hat, P_x, P_w, P_v, h, additive_noise):
    nx = x_hat.shape[0]
    iterations = z_true_all.shape[0]

    if not additive_noise and w_hat.shape[0] != nx:
        raise ValueError(
            "non-additive branch requires len(w_hat) == len(x_hat): the "
            "sigma-point slice for process noise uses nx as its width "
            "(matching the MATLAB source), not len(w_hat)")

    x_hat_all = np.zeros((nx, iterations))
    P_x = np.array(P_x, dtype=float)

    for k in range(iterations):
        # ---- Step 1a: build augmented sigma points ----
        if additive_noise:
            xa_hat = x_hat
            P_xa = P_x.copy()
        else:
            xa_hat = np.concatenate([x_hat, w_hat, v_hat])
            Pw_k = P_w[:, :, k] if P_w.ndim == 3 else P_w
            Pv_k = P_v[:, :, k] if P_v.ndim == 3 else P_v
            P_xa = _block_diag(P_x, Pw_k, Pv_k)

        P_xa = np.real(P_xa)
        if np.any(np.linalg.eigvalsh(0.5 * (P_xa + P_xa.T)) <= 0):
            P_xa = make_pd(P_xa, 1e-5, 1)

        Sigma_xa = np.linalg.cholesky(P_xa)
        L = xa_hat.shape[0]

        Wmx = np.empty(2 * L + 1)
        Wmx[0] = (h ** 2 - L) / h ** 2
        Wmx[1:] = 1.0 / (2 * h ** 2)
        Wcx = Wmx

        X_a_1 = xa_hat[:, None] + h * np.column_stack(
            [np.zeros(L), Sigma_xa, -Sigma_xa])

        # ---- Step 1a cont'd: propagate through dynamics ----
        if additive_noise:
            X_x_1 = X_a_1[:nx, :]
            sqrtP_w = np.linalg.cholesky(P_w)
            process_noise = sqrtP_w @ np.random.standard_normal(X_x_1.shape)
            X_x_2 = dynamics_f(dt, X_x_1, process_noise)
        else:
            X_x_1 = X_a_1[:nx, :]
            X_w_1 = X_a_1[nx:2 * nx, :]
            X_v_1 = X_a_1[2 * nx:, :]
            X_x_2 = dynamics_f(dt, X_x_1, X_w_1)

        x_hat = X_x_2 @ Wmx

        # ---- Step 1b: state covariance prediction ----
        X_diff_first = X_x_2[:, 0] - x_hat
        X_diff_rest = X_x_2[:, 1:] - x_hat[:, None]
        P_x = (Wcx[0] * np.outer(X_diff_first, X_diff_first)
                + (Wcx[1:] * X_diff_rest) @ X_diff_rest.T)

        # ---- Step 1c: output prediction ----
        if additive_noise:
            sqrtP_v = np.linalg.cholesky(P_v)
            measurement_noise = sqrtP_v @ np.random.standard_normal(X_x_2.shape)
            Z = measurement_f(X_x_2, measurement_noise)
        else:
            Z = measurement_f(X_x_2, X_v_1)

        z_hat = Z @ Wmx
        z_true = z_true_all[k, :]

        # ---- Step 2a: Kalman gain ----
        Z_diff_first = Z[:, 0] - z_hat
        Z_diff_rest = Z[:, 1:] - z_hat[:, None]
        Sigma_z = (Wcx[0] * np.outer(Z_diff_first, Z_diff_first)
                    + (Wcx[1:] * Z_diff_rest) @ Z_diff_rest.T)
        Sigma_xz = (Wcx[0] * np.outer(X_diff_first, Z_diff_first)
                     + (Wcx[1:] * X_diff_rest) @ Z_diff_rest.T)
        K_x = np.linalg.solve(Sigma_z.T, Sigma_xz.T).T  # Sigma_xz / Sigma_z

        # ---- Step 2b/2c: state and covariance update ----
        x_hat = x_hat + K_x @ (z_true - z_hat)
        P_x = P_x - K_x @ Sigma_z @ K_x.T
        P_x = 0.5 * (P_x + P_x.T)  # guard against float drift, see docstring

        x_hat_all[:, k] = x_hat

    return x_hat_all, P_x
