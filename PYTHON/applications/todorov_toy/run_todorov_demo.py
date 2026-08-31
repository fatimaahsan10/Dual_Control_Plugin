"""
Runnable demo of ilqg.py on a simple linear system, matching the setup
of Todorov's demo_linear.m: a random marginally-stable linear system
with quadratic running/terminal cost (cost = 0.5 x'Qx + 0.5 u'Ru).

See test_ilqg.py for the same system used as an automated regression
test against an independently-derived LQR solution; this script is the
"run it and look at the numbers" version of that.
"""

import numpy as np
from scipy.linalg import expm

from core.ddp_solver.ilqg import ilqg
from core.ddp_solver.forward_pass import forward_pass


def make_system(rng, n, m, h):
    A = rng.standard_normal((n, n))
    A = A - A.T                 # skew-symmetric -> purely imaginary eigenvalues
    A = expm(h * A)             # discrete time, marginally stable
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


def main():
    rng = np.random.default_rng(0)
    n, m, N, h = 3, 2, 20, 0.05

    A, B, Q, R = make_system(rng, n, m, h)
    step_fn, derivs_fn = make_step_and_derivs(A, B, Q, R)

    x0 = rng.standard_normal(n)
    u0 = 0.1 * rng.standard_normal((m, N))

    # cost of the initial (un-optimized) trajectory, for comparison
    x_init, _, cost_init = forward_pass(
        x0, u0, None, None, None, np.array([1.0]), step_fn, None
    )
    x_init = x_init[:, :, 0]
    cost_init_total = cost_init[:, 0].sum()

    print("=" * 60)
    print("Todorov iLQG demo -- simple linear system (demo_linear.m analog)")
    print("=" * 60)
    print(f"\nInitial state x0:\n{x0}")
    print(f"\nInitial control sequence u0 (shape {u0.shape}):\n{u0}")
    print(f"\nInitial trajectory x (shape {x_init.shape}):\n{x_init}")
    print(f"\nInitial total cost: {cost_init_total:.6f}")

    print("\n--- Running iLQG ---\n")
    x, u, L, Vx, Vxx, cost, trace, stop_reason = ilqg(
        step_fn, derivs_fn, x0, u0, lims=None, max_iter=50,
        tol_fun=1e-12, tol_grad=1e-10, verbose=1)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nStop reason: {stop_reason}")
    print(f"Iterations: {trace[-1]['iter']}")

    print(f"\nFinal optimized control sequence u (shape {u.shape}):\n{u}")
    print(f"\nFinal trajectory x (shape {x.shape}):\n{x}")
    print(f"\nFinal total cost: {cost.sum():.6f}")
    print(f"\nCost reduction: {cost_init_total:.6f} -> {cost.sum():.6f} "
          f"({100 * (1 - cost.sum() / cost_init_total):.2f}% lower)")


if __name__ == "__main__":
    main()
