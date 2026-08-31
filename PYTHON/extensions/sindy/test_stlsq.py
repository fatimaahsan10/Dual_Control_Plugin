"""Checks for stlsq.py against stlsq.m."""

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from extensions.sindy.stlsq import stlsq


def test_recovers_exact_sparse_ground_truth_noiseless():
    rng = np.random.default_rng(0)
    N, n_terms, n_states = 200, 8, 2
    Theta = rng.standard_normal((N, n_terms))

    Xi_true = np.zeros((n_terms, n_states))
    Xi_true[1, 0] = 2.5
    Xi_true[4, 0] = -1.8
    Xi_true[2, 1] = 3.0
    Xi_true[6, 1] = -2.2

    dX = Theta @ Xi_true  # exact, noiseless

    Xi = stlsq(Theta, dX, lam=0.3, max_iter=50, tol=1e-10, verbose=False)

    assert Xi.shape == (n_terms, n_states)
    recovered_support = (Xi != 0)
    true_support = (Xi_true != 0)
    assert np.array_equal(recovered_support, true_support)
    assert np.allclose(Xi, Xi_true, atol=1e-3)


def test_protected_term_survives_aggressive_threshold_unprotected_does_not():
    rng = np.random.default_rng(1)
    N, n_terms = 300, 5
    Theta = rng.standard_normal((N, n_terms))

    # index 0: large, legitimately significant coefficient
    # index 2: tiny coefficient, but PROTECTED -> must survive
    # index 3: equally tiny coefficient, unprotected -> must be zeroed
    Xi_true = np.array([[3.0], [0.0], [0.01], [0.01], [0.0]])
    dX = Theta @ Xi_true  # exact, noiseless

    protected_idx = np.array([False, False, True, False, False])

    Xi = stlsq(Theta, dX, lam=0.05, max_iter=50, tol=1e-10, verbose=False,
                protected_idx=protected_idx)

    assert Xi[0, 0] != 0.0
    assert Xi[2, 0] != 0.0   # protected: survives despite tiny true coefficient
    assert Xi[3, 0] == 0.0   # unprotected: thresholded away


def test_max_terms_cap_prunes_smallest_unprotected_first():
    rng = np.random.default_rng(2)
    N, n_terms = 300, 6
    Theta = rng.standard_normal((N, n_terms))

    # 5 well-separated true coefficients (index 5 is exactly 0), so a
    # tiny lambda alone won't prune anything -- only the max_terms cap
    # will. index 4 has the SMALLEST true magnitude but is protected,
    # to check protection overrides magnitude-based cap pruning.
    Xi_true = np.array([[3.0], [2.5], [2.0], [1.5], [1.0], [0.0]])
    dX = Theta @ Xi_true

    protected_idx = np.array([False, False, False, False, True, False])

    Xi = stlsq(Theta, dX, lam=0.01, max_iter=50, tol=1e-10, verbose=False,
                max_terms=3, protected_idx=protected_idx)

    nonzero = np.where(Xi[:, 0] != 0)[0]
    assert nonzero.size == 3
    assert 4 in nonzero  # protected term must survive the cap
    # cap prunes smallest-magnitude unprotected first -> the two
    # largest-magnitude unprotected terms (0, 1) should survive
    assert set(nonzero) == {0, 1, 4}


def test_min_terms_floor_revives_highest_magnitude():
    rng = np.random.default_rng(3)
    N, n_terms = 300, 5
    Theta = rng.standard_normal((N, n_terms))

    Xi_true = np.array([[3.0], [2.5], [2.0], [1.5], [1.0]])
    dX = Theta @ Xi_true

    # lambda far above every normalised coefficient's magnitude -> the
    # thresholding step alone would zero everything; min_terms=2 forces
    # a revival of the two highest-|initial-fit| terms.
    Xi = stlsq(Theta, dX, lam=1.0, max_iter=50, tol=1e-10, verbose=False,
                min_terms=2)

    nonzero = np.where(Xi[:, 0] != 0)[0]
    assert nonzero.size == 2
    assert set(nonzero) == {0, 1}  # two largest true coefficients
    assert np.sign(Xi[0, 0]) == np.sign(Xi_true[0, 0])
    assert np.sign(Xi[1, 0]) == np.sign(Xi_true[1, 0])


def test_composes_with_build_sindy_library_at_katies_real_dims():
    n_states, n_actions, n_sessions, n_patients = 6, 13, 12, 30
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    cfg = {"poly_order": 2, "include_cross": True, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, cfg)

    Xi = stlsq(Theta, dX, lam=0.05, max_iter=50, tol=1e-6, verbose=False,
                max_terms=22, protected_idx=protected_idx, min_terms=3)

    n_terms = Theta.shape[1]
    assert Xi.shape == (n_terms, n_states)
    assert np.all(np.isfinite(Xi))
    # default-protected action terms (a1..a13) must never be zeroed
    protected_rows = Xi[protected_idx, :]
    assert np.all(np.count_nonzero(protected_rows, axis=0) == protected_rows.shape[0])
    for s in range(n_states):
        assert np.count_nonzero(Xi[:, s]) <= 22


if __name__ == "__main__":
    tests = [test_recovers_exact_sparse_ground_truth_noiseless,
              test_protected_term_survives_aggressive_threshold_unprotected_does_not,
              test_max_terms_cap_prunes_smallest_unprotected_first,
              test_min_terms_floor_revives_highest_magnitude,
              test_composes_with_build_sindy_library_at_katies_real_dims]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
