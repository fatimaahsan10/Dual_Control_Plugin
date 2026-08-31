"""
End-to-end check of ilqg.py (+ forward_pass.py + back_pass.py + box_qp.py)
against an independently-computed finite-horizon discrete LQR solution.

For a linear system with quadratic cost, iLQG's local quadratic model is
*exact* -- so a correct implementation should converge to the same
solution as directly solving the Riccati recursion, in very few
iterations. This exercises the whole pipeline at once, the way
demo_linear.m does, just without plotting and with independently-derived
ground truth instead of "it looks converged".
"""

import numpy as np
from scipy.linalg import expm

from core.ddp_solver.ilqg import ilqg

# same convention as demo_linear.m's lin_dyn_cst: cost = 0.5 x'Qx + 0.5 u'Ru,
# same Q used for running and terminal cost, u is zeroed for the final
# (control-free) cost evaluation.


def make_system(rng, n, m, h):
    A = rng.standard_normal((n, n))
    A = A - A.T                 # skew-symmetric -> purely imaginary eigenvalues
    A = expm(h * A)             # discrete time, marginally stable (matches demo_linear.m)
    B = h * rng.standard_normal((n, m))
    Q = h * np.eye(n)
    R = 0.1 * h * np.eye(m)
    return A, B, Q, R


def make_step_and_derivs(A, B, Q, R):
    def step_fn(x, u, i):
        u = np.where(np.isnan(u), 0.0, u)
        x_next = A @ x + B @ u
        cost = 0.5 * np.sum(x * (Q @ x), axis=0) + 0.5 * np.sum(u * (R @ u), axis=0)
        return x_next, cost

    def derivs_fn(x, u_aug):
        n_time = x.shape[1]
        u0 = np.where(np.isnan(u_aug), 0.0, u_aug)
        fx = np.repeat(A[:, :, None], n_time, axis=2)
        fu = np.repeat(B[:, :, None], n_time, axis=2)
        cx = Q @ x
        cu = R @ u0
        cxx = np.repeat(Q[:, :, None], n_time, axis=2)
        cxu = np.zeros((A.shape[0], B.shape[1], n_time))
        cuu = np.repeat(R[:, :, None], n_time, axis=2)
        return fx, fu, None, None, None, cx, cu, cxx, cxu, cuu

    return step_fn, derivs_fn


def lqr_riccati(A, B, Q, R, N):
    """Standard finite-horizon discrete LQR, terminal weight = Q."""
    P = Q.copy()
    gains = [None] * N
    for k in range(N - 1, -1, -1):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        gains[k] = K
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
    return P, gains  # P here is P_0


def test_matches_lqr_unconstrained():
    rng = np.random.default_rng(0)
    n, m, N, h = 3, 2, 20, 0.05
    A, B, Q, R = make_system(rng, n, m, h)
    step_fn, derivs_fn = make_step_and_derivs(A, B, Q, R)

    x0 = rng.standard_normal(n)
    u0 = 0.1 * rng.standard_normal((m, N))

    P0, gains = lqr_riccati(A, B, Q, R, N)
    optimal_cost = 0.5 * x0 @ P0 @ x0

    x, u, L, Vx, Vxx, cost, trace, stop_reason = ilqg(
        step_fn, derivs_fn, x0, u0, lims=None, max_iter=50,
        tol_fun=1e-12, tol_grad=1e-10, verbose=0)

    assert stop_reason.startswith("SUCCESS"), stop_reason
    assert np.isclose(cost.sum(), optimal_cost, rtol=1e-4, atol=1e-8), (
        f"ilqg cost {cost.sum()} vs LQR optimal {optimal_cost}"
    )

    # feedback gains should match the LQR gains (u = u_bar + k + L(x-x_bar),
    # and at the optimum this local law should equal -K_lqr @ x)
    for k in range(N):
        assert np.allclose(L[:, :, k], -gains[k], atol=1e-4), (
            f"gain mismatch at step {k}"
        )

    # should converge fast: the local quadratic model is exact for an LQ problem
    # (the last few iterations are typically tiny tolFun-chasing steps)
    assert len(trace) <= 15, f"expected fast convergence, took {len(trace)} iterations"


def test_respects_control_limits():
    rng = np.random.default_rng(1)
    n, m, N, h = 3, 2, 20, 0.05
    A, B, Q, R = make_system(rng, n, m, h)
    step_fn, derivs_fn = make_step_and_derivs(A, B, Q, R)

    x0 = 3.0 * rng.standard_normal(n)  # larger x0 -> unconstrained optimum likely exceeds bounds
    u0 = np.zeros((m, N))

    P0, _ = lqr_riccati(A, B, Q, R, N)
    unconstrained_cost = 0.5 * x0 @ P0 @ x0

    lims = np.array([[-0.6, 0.6], [-0.6, 0.6]])

    x, u, L, Vx, Vxx, cost, trace, stop_reason = ilqg(
        step_fn, derivs_fn, x0, u0, lims=lims, max_iter=100,
        tol_fun=1e-10, tol_grad=1e-8, verbose=0)

    assert np.all(u >= lims[:, 0:1] - 1e-8) and np.all(u <= lims[:, 1:2] + 1e-8)
    # a tighter feasible set can't beat the unconstrained optimum
    assert cost.sum() >= unconstrained_cost - 1e-6


if __name__ == "__main__":
    tests = [test_matches_lqr_unconstrained, test_respects_control_limits]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
