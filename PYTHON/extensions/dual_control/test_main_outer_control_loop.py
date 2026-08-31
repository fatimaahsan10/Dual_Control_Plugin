"""
Integration check for main_outer_control_loop.py -- every piece it wires
together (ilqg_function, simulate_system, spkf_function) was already
verified in isolation, so this checks the WIRING: does the whole MPC loop
run without crashing, stay numerically sane, and -- since the test system
starts with a deliberately WRONG parameter estimate -- does the adaptive
machinery actually move the parameter estimate toward the truth (the
qualitative behaviour the whole dual-control apparatus exists to produce)?
"""

import numpy as np

from extensions.dual_control.main_outer_control_loop import main_outer_control_loop


def make_system(rng, nx, nu, n_p, ny):
    A = 0.9 * np.eye(nx) + 0.03 * rng.standard_normal((nx, nx))
    B0 = 0.5 * rng.standard_normal((nx, nu))
    F0 = np.eye(ny, nx)
    Q = np.eye(nx)
    R = 0.1 * np.eye(nu)

    def dynamics(dt, xa, u, constants, w, noise_index, augment_states,
                  w_from_filter, lam, u_lims, u_lim_method):
        K = xa.shape[1]
        if u.shape[1] == 1 and K > 1:
            u = np.tile(u, (1, K))
        x = xa[:nx, :]
        if augment_states:
            p = xa[nx:nx + n_p, :]
        else:
            p = np.ones((n_p, K))

        if noise_index is False or noise_index == 0:
            x_next = A @ x + p[0:1, :] * (B0 @ u)
            xa_next = np.vstack([x_next, xa[nx:, :]]) if augment_states else x_next
            if w_from_filter:
                xa_next = xa_next + w
            return xa_next
        else:
            nxa = xa.shape[0]
            col = np.zeros((nxa, K))
            col[noise_index - 1, :] = 0.02
            return col

    def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                      v_from_filter):
        K = xa.shape[1]
        x = xa[:nx, :]
        if noise_index is False or noise_index == 0:
            y = F0 @ x
            if v_from_filter:
                y = y + v
            return y
        else:
            col = np.zeros((ny, K))
            col[noise_index - 1, :] = 0.02
            return col

    def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
        x = xa[:nx, :]
        u = np.nan_to_num(u)
        return 0.5 * np.sum(x * (Q @ x), axis=0) + 0.5 * np.sum(u * (R @ u), axis=0)

    def continuous_dynamics(dt, xa, u, constants, augment_states, u_lims,
                              u_lim_method):
        nxa = xa.shape[0]
        fa = np.zeros((nxa, xa.shape[1]))       # unused by main_outer_control_loop
        F = np.eye(nx)                           # unused by main_outer_control_loop
        Fp = np.array([[0.01]])                  # small parameter-drift noise
        return fa, nx, n_p, F, Fp

    return A, B0, dynamics, measurement, cost, continuous_dynamics


def test_outer_loop_runs_and_adapts_parameter_estimate():
    rng = np.random.default_rng(0)
    nx, nu, n_p, ny, nv = 2, 1, 1, 2, 2

    A, B0, dynamics, measurement, cost, continuous_dynamics = make_system(
        rng, nx, nu, n_p, ny)

    x_hat_0 = np.array([1.0, -0.5])
    x_true_0 = x_hat_0.copy()
    p_true = np.array([1.0])    # true control-effectiveness scale
    p_hat_0 = np.array([0.4])   # deliberately wrong initial guess
    cov_X = 0.1
    cov_P = 0.2 * np.eye(n_p)
    constants = np.zeros(0)
    u_lims = np.array([[-2.0, 2.0]])

    result = main_outer_control_loop(
        T=5, dt=1, x_hat_0=x_hat_0, x_true_0=x_true_0, p_hat_0=p_hat_0,
        p_true=p_true, cov_X=cov_X, cov_P=cov_P, constants=constants,
        u_lims=u_lims, u_lim_method=2, dynamics=dynamics,
        measurement=measurement, cost=cost,
        continuous_dynamics=continuous_dynamics, ny=ny, nv=nv,
        max_du_iterations=30, first_run_max_du_iterations=50,
        verbose=False)

    N = 5
    assert result["x_hat"].shape == (nx, N)
    assert result["p_hat"].shape == (n_p, N)
    assert result["x_true"].shape == (nx, N)
    assert result["u"].shape == (nu, N - 1)
    assert result["cost_true"].shape == (N,)
    assert result["cost_est"].shape == (N,)

    for key in ("x_hat", "p_hat", "x_true", "u", "cost_true", "cost_est"):
        assert np.all(np.isfinite(result[key])), f"{key} has non-finite values"

    # the true state shouldn't blow up under control starting from a wrong
    # parameter estimate
    assert np.all(np.abs(result["x_true"]) < 100)

    # the whole point of the dual-control machinery: the parameter estimate
    # should move toward the truth, not stay stuck or diverge
    err_initial = abs(p_hat_0[0] - p_true[0])
    err_final = abs(result["p_hat"][0, -1] - p_true[0])
    assert err_final < err_initial, (
        f"parameter estimate did not improve: {p_hat_0[0]} -> "
        f"{result['p_hat'][0, -1]} (true={p_true[0]})"
    )


if __name__ == "__main__":
    tests = [test_outer_loop_runs_and_adapts_parameter_estimate]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
