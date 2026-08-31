"""
Confirms todorov_estimator.py and ilqg_function.py -- ported from Andrew's
CODES/ANDREW'S CODE/Todorov_estimator.m and iLQG_function.m -- work
unchanged at Katie's problem dimensions (6 states, 13 actions), same
approach as test_core_solver_dims.py (backward_pass.py / forward_pass.py).

Diff check (Compare-Object, Andrew's vs Katie's .m files): both
Todorov_estimator.m and iLQG_function.m differ from Andrew's only in
comments -- dead commented-out code and header attribution lines, no
logic changes. Confirmed no re-port needed on the MATLAB side.

Neither Python port hardcodes a dimension: todorov_estimator.py reads
nx/ny/nu/nw/nv off input array shapes, and ilqg_function.py reads
nu/N/nx/n_p off u_bar/x0/p_hat shapes. simulate_system.py (used by
ilqg_function.py for the nominal rollout and line search) is likewise
shape-generic. So this file imports Andrew's modules directly (no
duplication) and re-runs cut-down versions of andrew/test_todorov_estimator.py's
and andrew/test_ilqg_function.py's checks at n=6/m=13.

Katie's actual SCI plant isn't ported yet (see CLAUDE.md), so this uses
synthetic linear/quadratic stand-ins, same as test_core_solver_dims.py.
"""

import numpy as np

from extensions.dual_control.ilqg_function import ilqg_function
from extensions.dual_control.simulate_system import simulate_system
from extensions.dual_control.todorov_estimator import todorov_estimator

# Katie's problem dimensions (SCIM, BBS, AIS, Age, DPI, Caregiver support;
# 13-therapy WHO Package of Interventions action space).
N_STATES = 6
N_ACTIONS = 13


def make_synthetic_estimator_system(rng, nx, nu, ny, nw, nv, N):
    A0 = 0.5 * rng.standard_normal((nx, nx))
    B0 = 0.3 * rng.standard_normal((nx, nu))
    F0 = 0.4 * rng.standard_normal((ny, nx))
    E0 = 0.2 * rng.standard_normal((ny, nu))

    A = np.repeat(A0[:, :, None], N, axis=2)
    B = np.repeat(B0[:, :, None], N, axis=2)
    F = np.repeat(F0[:, :, None], N, axis=2)
    E = np.repeat(E0[:, :, None], N, axis=2)

    c = 0.05 * rng.standard_normal((nx, nw, N))
    Cx = 0.05 * rng.standard_normal((nx, nx, N, nw))
    Cu = 0.05 * rng.standard_normal((nx, nu, N, nw))
    d = 0.05 * rng.standard_normal((ny, nv, N))
    Dx = 0.05 * rng.standard_normal((ny, nx, N, nv))
    Du = 0.05 * rng.standard_normal((ny, nu, N, nv))

    l = 0.1 * rng.standard_normal((nu, N - 1))
    L = 0.1 * rng.standard_normal((nu, nx, N - 1))

    x_bar = 0.1 * rng.standard_normal((nx, N))
    y_bar = 0.1 * rng.standard_normal((ny, N))
    u_bar = 0.1 * rng.standard_normal((nu, N - 1))

    def measurement(dt, xa, u, constants, v, noise_index, augment_states, v_from_filter):
        return F0 @ xa + E0 @ u + v

    return A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar, u_bar, measurement


def run_estimator_at_katie_dims(seed):
    rng = np.random.default_rng(seed)
    # nv must equal ny here: the synthetic `measurement` below adds the
    # noise vector directly to F0@xa+E0@u (same simplification Andrew's
    # own test_todorov_estimator.py uses, where nv==ny==2 coincidentally).
    nx, nu, ny, nw, nv, N = N_STATES, N_ACTIONS, N_STATES, 2, N_STATES, 6
    (A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar, u_bar,
     measurement) = make_synthetic_estimator_system(rng, nx, nu, ny, nw, nv, N)

    x_hat_0 = 0.1 * rng.standard_normal(nx)
    cov_x_hat_0 = np.eye(nx) * 0.5
    sqrtR = np.eye(nv)

    (x_hat, pi_ctrl, K, m_x, m_e, Sigma_x, Sigma_e, Sigma_xe, Sigma_ex,
     internals) = todorov_estimator(
        A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar, u_bar,
        x_hat_0, cov_x_hat_0, dt=1.0, constants=None, sqrtR=sqrtR,
        augment_states=True, measurement=measurement, return_internals=True)

    return dict(A=A, B=B, F=F, E=E, l=l, L=L, x_hat_0=x_hat_0, N=N, nx=nx,
                 nu=nu, ny=ny, x_hat=x_hat, pi_ctrl=pi_ctrl, K=K, m_x=m_x,
                 m_e=m_e, Sigma_x=Sigma_x, Sigma_e=Sigma_e, Sigma_xe=Sigma_xe,
                 Sigma_ex=Sigma_ex, P=internals["P"], M=internals["M"])


def test_estimator_shapes_and_gain_equation_at_6state_13action():
    ctx = run_estimator_at_katie_dims(seed=0)
    nx, nu, ny, N = ctx["nx"], ctx["nu"], ctx["ny"], ctx["N"]

    assert ctx["K"].shape == (nx, ny, N - 1)
    assert ctx["pi_ctrl"].shape == (nu, N - 1)
    assert ctx["x_hat"].shape == (nx, N)

    # m_e (mean estimation error) is homogeneous with zero forcing/initial
    # condition -- must stay identically 0 regardless of dimension.
    assert np.allclose(ctx["m_e"], 0.0, atol=1e-10)

    # K_k is defined by K_k (F Sigma_e F' + P_k) = A Sigma_e F' (eq. 2.36)
    K, F, A, Sigma_e, P = ctx["K"], ctx["F"], ctx["A"], ctx["Sigma_e"], ctx["P"]
    for k in range(N - 1):
        Fk, Ak, Se, Pk = F[:, :, k], A[:, :, k], Sigma_e[:, :, k], P[:, :, k]
        lhs = K[:, :, k] @ (Fk @ Se @ Fk.T + Pk)
        rhs = Ak @ Se @ Fk.T
        assert np.allclose(lhs, rhs, atol=1e-8), f"gain equation violated at k={k}"


def make_linear_system_katie_dims(rng, nx, nu, ny, sigma_w=0.05, sigma_v=0.05):
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


def test_ilqg_function_converges_and_matches_lqr_at_6state_13action():
    rng = np.random.default_rng(1)
    nx, nu, ny, N = N_STATES, N_ACTIONS, N_STATES, 5
    dt = 1.0  # Katie's session interval (1 week)
    nw, nv = nx, ny

    A, B, F, E, Q, R, dynamics, measurement, cost = make_linear_system_katie_dims(
        rng, nx, nu, ny)

    x0 = rng.standard_normal(nx)
    p_hat = np.zeros(0)
    cov_xa_hat_0 = 1e-2 * np.eye(nx)
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

    assert xa0.shape == (nx, N + 1)
    assert u_bar.shape == (nu, N)
    assert l.shape == (nu, N)
    assert L.shape == (nu, nx, N)
    assert Pw.shape == (nx, nx, N)
    assert Pv.shape == (ny, ny, N)
    assert np.all(np.isfinite(Pw)) and np.all(np.isfinite(Pv))

    P0 = lqr_riccati(A, B, Q, R, N)
    optimal_cost = 0.5 * x0 @ P0 @ x0
    assert np.isclose(cost_arr.sum(), optimal_cost, rtol=1e-6), (
        f"ilqg_function cost {cost_arr.sum()} vs LQR optimal {optimal_cost}"
    )


if __name__ == "__main__":
    tests = [test_estimator_shapes_and_gain_equation_at_6state_13action,
              test_ilqg_function_converges_and_matches_lqr_at_6state_13action]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
