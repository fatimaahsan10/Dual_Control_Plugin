"""
Sanity checks for box_qp.py against known solutions and against
scipy.optimize.minimize as an independent solver.

Run directly: python test_box_qp.py
(also collectible by pytest, since functions are named test_*)
"""

import numpy as np
from scipy.optimize import minimize

from core.ddp_solver.box_qp import box_qp


def random_pd(n, rng, cond_boost=1.0):
    A = rng.standard_normal((n, n))
    return A @ A.T + cond_boost * np.eye(n)


def test_unconstrained_interior():
    # bounds are wide enough that the true optimum -H^-1 g is inside the
    # box -> box_qp should behave like plain Newton, all dims free.
    rng = np.random.default_rng(0)
    n = 4
    H = random_pd(n, rng)
    g = rng.standard_normal(n)
    x_star = np.linalg.solve(H, -g)

    lower = x_star - 10.0
    upper = x_star + 10.0

    x, result, Hfree, free = box_qp(H, g, lower, upper)

    assert result >= 1, f"expected success, got result={result}"
    assert np.all(free), "expected all dimensions free with wide bounds"
    assert np.allclose(x, x_star, atol=1e-6), (x, x_star)


def test_hfree_matches_matlab_chol_convention():
    # back_pass.py consumes Hfree as upper-triangular with Hfree.T @ Hfree
    # == H[free,free] (MATLAB chol() convention) -- lock that in.
    rng = np.random.default_rng(1)
    n = 5
    H = random_pd(n, rng)
    g = rng.standard_normal(n)
    lower = np.full(n, -10.0)
    upper = np.full(n, 10.0)

    x, result, Hfree, free = box_qp(H, g, lower, upper)

    assert result >= 1
    Hff = H[np.ix_(free, free)]
    assert np.allclose(Hfree.T @ Hfree, Hff, atol=1e-8)
    # upper-triangular check
    assert np.allclose(Hfree, np.triu(Hfree))


def test_all_dims_pushed_to_bounds():
    # g strongly rewards decreasing every x_i, box is tight -> every
    # dimension should clamp at its lower bound.
    n = 3
    H = np.eye(n)
    g = np.array([10.0, 10.0, 10.0])  # unconstrained optimum is x = -g (very negative)
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])

    x, result, Hfree, free = box_qp(H, g, lower, upper)

    assert result == 6, f"expected all-clamped result=6, got {result}"
    assert np.allclose(x, lower, atol=1e-8)
    assert not np.any(free)


def test_feasibility_and_matches_scipy_random():
    rng = np.random.default_rng(42)
    for trial in range(10):
        n = rng.integers(2, 6)
        H = random_pd(n, rng, cond_boost=1.0)
        g = rng.standard_normal(n)
        lower = -rng.uniform(0.5, 2.0, size=n)
        upper = rng.uniform(0.5, 2.0, size=n)

        x, result, Hfree, free = box_qp(H, g, lower, upper)
        assert result >= 1, f"trial {trial}: box_qp failed, result={result}"

        # feasibility
        assert np.all(x >= lower - 1e-9) and np.all(x <= upper + 1e-9)

        # independent solver cross-check
        def objective(z):
            return 0.5 * z @ H @ z + g @ z, H @ z + g

        res = minimize(objective, x0=np.zeros(n), jac=True, method="L-BFGS-B",
                        bounds=list(zip(lower, upper)))

        box_qp_value = 0.5 * x @ H @ x + g @ x
        assert np.isclose(box_qp_value, res.fun, atol=1e-5, rtol=1e-4), (
            f"trial {trial}: box_qp value {box_qp_value} vs scipy {res.fun}"
        )


if __name__ == "__main__":
    tests = [
        test_unconstrained_interior,
        test_hfree_matches_matlab_chol_convention,
        test_all_dims_pushed_to_bounds,
        test_feasibility_and_matches_scipy_random,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
