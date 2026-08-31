"""
Integration checks for ilqg_function.py -- the math of each piece
(forward_pass, backward_pass, todorov_estimator) was already verified in
isolation; this checks that the WIRING between them is correct: does data
flow through correctly, does the whole loop converge, and (for a case with
small, purely-additive noise and small initial state uncertainty, so the
dual-control correction should be small) does the result approximately
match an independently-computed LQR optimum, the same style of check used
for Todorov's ilqg.py.
"""

import numpy as np

from extensions.dual_control.ilqg_function import ilqg_function
from extensions.dual_control.simulate_system import simulate_system


def make_linear_system(rng, nx, nu, ny, sigma_w=0.05, sigma_v=0.05):
    A = 0.9 * np.eye(nx) + 0.05 * rng.standard_normal((nx, nx))
    B = 0.3 * rng.standard_normal((nx, nu))
    F = np.eye(ny, nx)  # fully observed (ny == nx here)
    E = np.zeros((ny, nu))
    Q = np.eye(nx)
    R = np.eye(nu)

    def dynamics(dt, xa, u, constants, w, noise_index, augment_states,
                  w_from_filter, lam, u_lims, u_lim_method):
        if noise_index is False or noise_index == 0:
            return A @ xa + B @ u
        # small, x/u-independent (purely additive) noise column i -- keeps
        # Sigma_e away from numerical singularity without exercising the
        # multiplicative-noise Jacobian machinery (already covered by
        # test_forward_pass.py)
        col = np.zeros((nx, xa.shape[1]))
        col[noise_index - 1, :] = sigma_w
        return col

    def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                      v_from_filter):
        if noise_index is False or noise_index == 0:
            return F @ xa + E @ u
        col = np.zeros((ny, xa.shape[1]))
        col[noise_index - 1, :] = sigma_v
        return col

    def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
        u = np.nan_to_num(u)
        return 0.5 * np.sum(xa * (Q @ xa), axis=0) + 0.5 * np.sum(u * (R @ u), axis=0)

    return A, B, F, E, Q, R, dynamics, measurement, cost


def lqr_riccati(A, B, Q, R, N):
    P = Q.copy()
    for _ in range(N):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
    return P


def test_converges_and_approximately_matches_lqr():
    rng = np.random.default_rng(0)
    nx, nu, ny, N = 2, 1, 2, 5
    dt = 0.1
    nw, nv = nx, ny  # one small additive noise channel per state/measurement

    A, B, F, E, Q, R, dynamics, measurement, cost = make_linear_system(rng, nx, nu, ny)

    x0 = rng.standard_normal(nx)
    p_hat = np.zeros(0)
    cov_xa_hat_0 = 1e-2 * np.eye(nx)  # small but not pathologically tiny
    l0 = np.zeros((nu, N))
    L0 = np.zeros((nu, nx, N))
    u_bar0 = np.zeros((nu, N))

    xa0, u_bar, l, L, lam, dlambda, cost_arr, Pw, Pv, converged = ilqg_function(
        T=N * dt, dt=dt, x0=x0, l=l0, L=L0, u_bar=u_bar0, lam=1.0, dlambda=1.0,
        constants=np.zeros(0), p_hat=p_hat, cov_xa_hat_0=cov_xa_hat_0,
        augment_states=False, reg_type=1, u_lims=None, ny=ny, nv=nv, nw=nw,
        max_du_iterations=100, dyn_noise_reg=0.0, tracking_trajectory=None,
        u_lim_method=2, dynamics=dynamics, measurement=measurement, cost=cost,
        simulate_system_fn=simulate_system, verbose=False)

    assert converged in (1, 2), f"did not converge cleanly, converged={converged}"

    # shapes
    assert xa0.shape == (nx, N + 1)
    assert u_bar.shape == (nu, N)
    assert l.shape == (nu, N)
    assert L.shape == (nu, nx, N)
    assert Pw.shape == (nx, nx, N)
    assert Pv.shape == (ny, ny, N)
    assert np.all(np.isfinite(Pw)) and np.all(np.isfinite(Pv))

    # cross-check against an independently-computed LQR optimum. The
    # reported cost is evaluated on the nominal (zero-injected-noise)
    # certainty-equivalent rollout, which for a linear/quadratic system
    # should match the LQR solution almost exactly regardless of the
    # noise/covariance settings above -- empirically this matches to
    # ~1e-10 relative error, so a loose tolerance here would hide real
    # regressions.
    P0 = lqr_riccati(A, B, Q, R, N)
    optimal_cost = 0.5 * x0 @ P0 @ x0
    assert np.isclose(cost_arr.sum(), optimal_cost, rtol=1e-6), (
        f"ilqg_function cost {cost_arr.sum()} vs LQR optimal {optimal_cost}"
    )


if __name__ == "__main__":
    tests = [test_converges_and_approximately_matches_lqr]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
