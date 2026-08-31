"""Checks for compute_fisher_information.py against compute_fisher_information.m."""

import warnings

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from extensions.sindy.compute_fisher_information import compute_fisher_information


def test_matches_closed_form_formula():
    rng = np.random.default_rng(0)
    N, n_terms = 200, 6
    Theta = rng.standard_normal((N, n_terms))
    noise_std = 0.2

    F = compute_fisher_information(Theta, noise_std)

    col_std = Theta.std(axis=0, ddof=1)
    Theta_n = Theta / col_std
    expected = (1.0 / noise_std ** 2) * (Theta_n.T @ Theta_n)
    expected = (expected + expected.T) / 2.0

    assert F.shape == (n_terms, n_terms)
    assert np.allclose(F, expected, atol=1e-8)


def test_near_zero_noise_std_falls_back_to_1e6_with_warning():
    rng = np.random.default_rng(1)
    N, n_terms = 100, 4
    Theta = rng.standard_normal((N, n_terms))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        F = compute_fisher_information(Theta, noise_std=0.0)

    assert any("noise_std near zero" in str(w.message) for w in caught)

    col_std = Theta.std(axis=0, ddof=1)
    Theta_n = Theta / col_std
    expected = (1.0 / 1e-6) * (Theta_n.T @ Theta_n)
    expected = (expected + expected.T) / 2.0
    assert np.allclose(F, expected, atol=1e-2)  # loose tol: huge-magnitude entries


def test_symmetric_and_positive_semidefinite():
    rng = np.random.default_rng(2)
    N, n_terms = 150, 8
    Theta = rng.standard_normal((N, n_terms))

    F = compute_fisher_information(Theta, noise_std=0.1)

    assert np.array_equal(F, F.T)
    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals >= -1e-8)


def test_invariant_to_arbitrary_per_column_physical_unit_rescaling():
    # F is computed on NORMALISED Theta specifically so it doesn't
    # depend on arbitrary physical-unit choices per library term --
    # rescaling any column of Theta by a positive constant must leave
    # F unchanged, since normalising by that column's own std cancels
    # the rescaling exactly.
    rng = np.random.default_rng(3)
    N, n_terms = 200, 5
    Theta = rng.standard_normal((N, n_terms))
    rescale = rng.uniform(0.01, 100.0, size=n_terms)
    Theta_rescaled = Theta * rescale

    F1 = compute_fisher_information(Theta, noise_std=0.3)
    F2 = compute_fisher_information(Theta_rescaled, noise_std=0.3)

    assert np.allclose(F1, F2, atol=1e-6)


def test_composes_with_build_sindy_library_at_katies_real_dims():
    # poly_order=2+cross with only 20 patients (200 interior-session
    # observations) builds a Theta with 210 candidate terms -- MORE
    # terms than observations, so Theta.T @ Theta is genuinely
    # rank-deficient, not just poorly conditioned. This naturally
    # exercises the not-PSD regularisation branch (see module
    # docstring) rather than the closed-form/well-conditioned path the
    # other tests cover.
    n_states, n_actions, n_sessions, n_patients = 6, 13, 12, 20
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    cfg = {"poly_order": 2, "include_cross": True, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, cfg)
    assert Theta.shape[1] > Theta.shape[0], "test setup expects an underdetermined library"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        F = compute_fisher_information(Theta, noise_std=0.05)

    assert any("not positive semi-definite" in str(w.message) for w in caught)

    n_terms = Theta.shape[1]
    assert F.shape == (n_terms, n_terms)
    assert np.array_equal(F, F.T)
    assert np.all(np.isfinite(F))
    cond_number = np.linalg.cond(F)
    assert np.isfinite(cond_number)


if __name__ == "__main__":
    tests = [test_matches_closed_form_formula,
              test_near_zero_noise_std_falls_back_to_1e6_with_warning,
              test_symmetric_and_positive_semidefinite,
              test_invariant_to_arbitrary_per_column_physical_unit_rescaling,
              test_composes_with_build_sindy_library_at_katies_real_dims]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
