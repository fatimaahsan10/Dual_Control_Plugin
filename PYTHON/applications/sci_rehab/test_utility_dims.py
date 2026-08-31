"""
Quick standalone dimension checks for box_qp.py, make_pd.py, and
finite_difference.py at Katie's sizes. These three were already
exercised transitively (box_qp inside backward_pass.py's constrained
branch, make_pd inside spkf_function.py, finite_difference inside
forward_pass.py -- see test_core_solver_dims.py / test_spkf_function_dims.py)
but never driven directly at Katie's dimensions, and none of the three
has any prior standalone test at Katie's actual sizes (todorov/test_box_qp.py
and andrew/test_make_pd.py both test only at small generic n=3..5).

boxQP.m / makePD.m / finite_difference.m were already confirmed
functionally identical to Andrew's versions in CLAUDE.md's file-by-file
diff (0 substantive diff), and none of the three Python ports hardcodes
a dimension (n is read off H.shape[0] / x.shape[0] in each), so these
are quick confirmations rather than full re-derivations -- the existing
test suites already establish correctness of the algorithms themselves.
"""

import numpy as np

from core.ddp_solver.box_qp import box_qp
from extensions.dual_control.make_pd import make_pd
from extensions.dual_control.finite_difference import finite_difference

N_STATES = 6
N_ACTIONS = 13


def test_box_qp_at_13_action_dim():
    # box_qp solves over the control (action) dimension inside
    # backward_pass.py's constrained branch -> n = N_ACTIONS = 13.
    rng = np.random.default_rng(0)
    n = N_ACTIONS
    A = rng.standard_normal((n, n))
    H = A @ A.T + np.eye(n)  # SPD
    g = rng.standard_normal(n)
    x_star = np.linalg.solve(H, -g)

    lower = x_star - 10.0  # bounds wide enough that x_star is interior
    upper = x_star + 10.0

    x, result, Hfree, free = box_qp(H, g, lower, upper)

    assert result >= 1, f"expected success, got result={result}"
    assert x.shape == (n,)
    assert np.all(free), "expected all 13 dimensions free with wide bounds"
    assert np.allclose(x, x_star, atol=1e-6)


def test_make_pd_at_6_state_dim():
    # make_pd cleans up the state covariance inside spkf_function.py's
    # sigma-point update -> n = N_STATES = 6.
    rng = np.random.default_rng(1)
    n = N_STATES
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigvals = rng.uniform(-1.0, 2.0, size=n)  # deliberately includes negatives
    H = V @ np.diag(eigvals) @ V.T
    epsilon = 1e-5

    H_pd = make_pd(H, epsilon, method=1)

    assert H_pd.shape == (n, n)
    assert np.isclose(np.linalg.eigvalsh(H_pd).min(), epsilon, atol=1e-10)
    assert np.allclose(H_pd, H_pd.T)


def test_finite_difference_at_19_augmented_dim():
    # finite_difference.py differentiates over the augmented
    # [state; action] vector inside forward_pass.py ->
    # n = N_STATES + N_ACTIONS = 19.
    rng = np.random.default_rng(2)
    n, K = N_STATES + N_ACTIONS, 4
    Aff = rng.standard_normal((n, n))

    def fun(x):
        return Aff @ x

    x = rng.standard_normal((n, K))
    J = finite_difference(fun, x)

    assert J.shape == (n, n, K)
    for k in range(K):
        assert np.allclose(J[:, :, k], Aff, atol=1e-4)


if __name__ == "__main__":
    tests = [test_box_qp_at_13_action_dim,
              test_make_pd_at_6_state_dim,
              test_finite_difference_at_19_augmented_dim]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
