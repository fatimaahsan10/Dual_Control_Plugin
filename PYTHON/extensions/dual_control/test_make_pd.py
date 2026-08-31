"""Checks for make_pd.py against the exact properties each method promises."""

import numpy as np

from extensions.dual_control.make_pd import make_pd


def random_symmetric_with_negative_eig(rng, n):
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))  # random orthonormal basis
    eigvals = rng.uniform(-1.0, 2.0, size=n)  # deliberately includes negatives
    return V @ np.diag(eigvals) @ V.T, eigvals


def test_method1_shifts_uniformly_to_epsilon():
    rng = np.random.default_rng(0)
    n = 5
    H, eigvals = random_symmetric_with_negative_eig(rng, n)
    epsilon = 1e-5

    H_pd = make_pd(H, epsilon, method=1)

    assert np.isclose(np.linalg.eigvalsh(H_pd).min(), epsilon, atol=1e-10)
    # every eigenvalue should have moved by exactly the same amount
    shift = epsilon - eigvals.min()
    assert np.allclose(H_pd, H + shift * np.eye(n), atol=1e-10)
    assert np.allclose(H_pd, H_pd.T)


def test_method2_only_clips_small_eigenvalues():
    rng = np.random.default_rng(1)
    n = 5
    H, eigvals = random_symmetric_with_negative_eig(rng, n)
    epsilon = 1e-5

    H_pd = make_pd(H, epsilon, method=2)
    new_eigvals = np.sort(np.linalg.eigvalsh(H_pd))
    expected = np.sort(np.maximum(eigvals, epsilon))

    assert np.allclose(new_eigvals, expected, atol=1e-8)
    assert np.all(new_eigvals >= epsilon - 1e-12)
    assert np.allclose(H_pd, H_pd.T)


def test_method2_leaves_already_pd_matrix_unchanged():
    rng = np.random.default_rng(2)
    n = 4
    A = rng.standard_normal((n, n))
    H = A @ A.T + np.eye(n)  # already well clear of epsilon

    H_pd = make_pd(H, 1e-5, method=2)
    assert np.allclose(H_pd, H, atol=1e-10)


def test_invalid_method_raises():
    H = np.eye(3)
    try:
        make_pd(H, 1e-5, method=3)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [test_method1_shifts_uniformly_to_epsilon,
              test_method2_only_clips_small_eigenvalues,
              test_method2_leaves_already_pd_matrix_unchanged,
              test_invalid_method_raises]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
