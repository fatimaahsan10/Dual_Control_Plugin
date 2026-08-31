"""
Fixed-gain estimator for dual/adaptive iLQG under multiplicative noise.

Ported from Andrew's CODES/ANDREW'S CODE/Todorov_estimator.m.
Based on Li, W. & Todorov, E. (2007), and Andrew Mathis's PhD thesis
(NOTES/2024_Mathis_Sensinger Carretero_PhD Thesis.pdf), section 2.1.3.2,
eqs. (2.35)-(2.48).

Because dynamics/measurement noise are multiplicative (state/control
dependent, via Cx/Cu/Dx/Du from forward_pass.py), their effective
covariance at each step depends on the current joint distribution of the
estimate x-hat and the estimation error e = x - x-hat -- not just a fixed
sensor-noise constant like plain Kalman filtering. So instead of
propagating one error covariance, this tracks five coupled moments:

    m_x, Sigma_x   mean / covariance of the ESTIMATE x-hat (not of x,
                    despite the name -- matches the MATLAB source)
    m_e, Sigma_e   mean / covariance of the estimation ERROR e = x - x-hat
    Sigma_xe       cross-covariance between the estimate and the error

The filter gain K (eq. 2.36) is structurally a standard Kalman gain, but
built from the state-dependent P_k/M_k (eqs. 2.47/2.48) rather than a
constant. K is what backward_pass.py takes as an external input -- this
function is what produces it, closing the control/estimation loop.

DEVIATION FROM THE LITERAL MATLAB SOURCE: the MATLAB hardcodes a call to
the SIDARTHE-specific `Measurement` function. This port takes it as an
explicit `measurement` callable instead (same contract as forward_pass.py
and Measurement.m), so this file doesn't depend on porting the
SIDARTHE-specific plant, matching how forward_pass.py was already done.

NOTE: the "vs" noise fed to `measurement` below is always multiplied by 0
(`vs = 0*sqrtR@...`), exactly matching the MATLAB source -- despite
sampling a `y_data`, this is a deterministic rollout, not a Monte Carlo
draw. Genuine stochastic simulation happens in simulate_system.m
(not yet ported).
"""

import numpy as np


def todorov_estimator(A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar,
                        u_bar, x_hat_0, cov_x_hat_0, dt, constants, sqrtR,
                        augment_states, measurement, return_internals=False):
    nx = x_hat_0.shape[0]
    ny = y_bar.shape[0]
    nu = B.shape[1]
    nw = c.shape[1]
    nv = d.shape[1]
    N = c.shape[2]  # trajectory length, matches forward_pass.py's N

    x_hat = np.zeros((nx, N))
    m_x = np.zeros((nx, N))
    m_e = np.zeros((nx, N))          # stays identically 0 (see docstring below)
    Sigma_x = np.zeros((nx, nx, N))
    Sigma_e = np.zeros((nx, nx, N))
    Sigma_xe = np.zeros((nx, nx, N))
    Sigma_ex = np.zeros((nx, nx, N))
    P = np.zeros((ny, ny, N - 1))
    M = np.zeros((nx, nx, N - 1))
    K = np.zeros((nx, ny, N - 1))
    pi_ctrl = np.zeros((nu, N - 1))  # matches MATLAB's `pi` -- realized u deviation
    y = np.zeros((ny, N - 1))

    x_hat[:, 0] = x_hat_0
    m_x[:, 0] = x_hat_0
    # m_e[:, 0] stays 0 (eq. 2.40); since its recursion below is
    # homogeneous (no forcing term), m_e is identically 0 for all k --
    # a useful invariant to test against.
    Sigma_x[:, :, 0] = np.outer(x_hat_0, x_hat_0)
    Sigma_e[:, :, 0] = cov_x_hat_0
    Sigma_xe[:, :, 0] = np.zeros_like(cov_x_hat_0)
    Sigma_ex[:, :, 0] = Sigma_xe[:, :, 0].T

    for k in range(N - 1):
        mxme = m_x[:, k] + m_e[:, k]
        lLmx = l[:, k] + L[:, :, k] @ m_x[:, k]
        Sigma_x_ex = Sigma_x[:, :, k] + Sigma_ex[:, :, k]
        Sigma_x_xe = Sigma_x[:, :, k] + Sigma_xe[:, :, k]
        all_sigmas = (Sigma_x[:, :, k] + Sigma_xe[:, :, k]
                       + Sigma_ex[:, :, k] + Sigma_e[:, :, k])
        l_uu = (np.outer(l[:, k], l[:, k])
                + np.outer(l[:, k], m_x[:, k]) @ L[:, :, k].T
                + L[:, :, k] @ np.outer(m_x[:, k], l[:, k])
                + L[:, :, k] @ Sigma_x[:, :, k] @ L[:, :, k].T)

        # ---- P_k: effective measurement noise covariance (eq. 2.47) ----
        for j in range(nv):
            dj, Dxj, Duj = d[:, j, k], Dx[:, :, k, j], Du[:, :, k, j]
            P[:, :, k] += (
                np.outer(dj, dj)
                + np.outer(dj, mxme) @ Dxj.T
                + Dxj @ np.outer(mxme, dj)
                + np.outer(dj, lLmx) @ Duj.T
                + Duj @ np.outer(lLmx, dj)
                + Dxj @ (np.outer(mxme, l[:, k]) + Sigma_x_ex @ L[:, :, k].T) @ Duj.T
                + Duj @ (np.outer(l[:, k], mxme) + L[:, :, k] @ Sigma_x_xe) @ Dxj.T
                + Dxj @ all_sigmas @ Dxj.T
                + Duj @ l_uu @ Duj.T)

        # ---- M_k: effective dynamics noise covariance (eq. 2.48) ----
        for i in range(nw):
            ci, Cxi, Cui = c[:, i, k], Cx[:, :, k, i], Cu[:, :, k, i]
            M[:, :, k] += (
                np.outer(ci, ci)
                + np.outer(ci, mxme) @ Cxi.T
                + Cxi @ np.outer(mxme, ci)
                + np.outer(ci, lLmx) @ Cui.T
                + Cui @ np.outer(lLmx, ci)
                + Cxi @ (np.outer(mxme, l[:, k]) + Sigma_x_ex @ L[:, :, k].T) @ Cui.T
                + Cui @ (np.outer(l[:, k], mxme) + L[:, :, k] @ Sigma_x_xe) @ Cxi.T
                + Cxi @ all_sigmas @ Cxi.T
                + Cui @ l_uu @ Cui.T)

        Ak, Bk, Fk = A[:, :, k], B[:, :, k], F[:, :, k]

        # ---- filter gain (eq. 2.36) ----
        denom = Fk @ Sigma_e[:, :, k] @ Fk.T + P[:, :, k]
        numer = Ak @ Sigma_e[:, :, k] @ Fk.T
        K[:, :, k] = np.linalg.solve(denom.T, numer.T).T  # matches MATLAB's numer/denom

        pi_ctrl[:, k] = l[:, k] + L[:, :, k] @ x_hat[:, k]

        # ---- deterministic "measurement" rollout (see module docstring) ----
        x_data = (x_bar[:, k] + x_hat[:, k]).reshape(-1, 1)
        u_data = (u_bar[:, k] + pi_ctrl[:, k]).reshape(-1, 1)
        vs = (0.0 * sqrtR @ np.random.standard_normal(nv)).reshape(-1, 1)
        y_data = measurement(dt, x_data, u_data, constants, vs, False,
                               augment_states, 0).reshape(-1)
        y[:, k] = y_data - y_bar[:, k]

        # ---- next-step state-estimate rollout (eq. 2.35) ----
        x_hat[:, k + 1] = (Ak @ x_hat[:, k] + Bk @ pi_ctrl[:, k]
                            + K[:, :, k] @ (y[:, k] - Fk @ x_hat[:, k]
                                             - E[:, :, k] @ pi_ctrl[:, k]))

        ABL = Ak + Bk @ L[:, :, k]
        KF = K[:, :, k] @ Fk
        AKF = Ak - KF

        # ---- moment propagation (eqs. 2.37, 2.39, 2.41, 2.43, 2.45) ----
        m_x[:, k + 1] = ABL @ m_x[:, k] + KF @ m_e[:, k] + Bk @ l[:, k]
        m_e[:, k + 1] = AKF @ m_e[:, k]

        mean_term = ABL @ m_x[:, k] + KF @ m_e[:, k]
        Sigma_x[:, :, k + 1] = (
            ABL @ Sigma_x[:, :, k] @ ABL.T + KF @ Sigma_e[:, :, k] @ Ak.T
            + ABL @ Sigma_xe[:, :, k] @ Fk.T @ K[:, :, k].T
            + KF @ Sigma_ex[:, :, k] @ ABL.T
            + np.outer(mean_term, l[:, k]) @ Bk.T
            + Bk @ np.outer(l[:, k], mean_term)
            + Bk @ np.outer(l[:, k], l[:, k]) @ Bk.T)

        # NOTE: (A - KF) on the left, plain A on the right -- not a typo,
        # matches thesis eq. (2.43) exactly. Sigma_e is not guaranteed
        # symmetric by this recursion; see test_todorov_estimator.py.
        Sigma_e[:, :, k + 1] = AKF @ Sigma_e[:, :, k] @ Ak.T + M[:, :, k]

        Sigma_xe[:, :, k + 1] = (ABL @ Sigma_xe[:, :, k] @ AKF.T
                                   + Bk @ np.outer(l[:, k], m_e[:, k]) @ AKF.T)
        Sigma_ex[:, :, k + 1] = Sigma_xe[:, :, k + 1].T

    if not return_internals:
        return x_hat, pi_ctrl, K, m_x, m_e, Sigma_x, Sigma_e, Sigma_xe, Sigma_ex

    return (x_hat, pi_ctrl, K, m_x, m_e, Sigma_x, Sigma_e, Sigma_xe, Sigma_ex,
            dict(P=P, M=M))
