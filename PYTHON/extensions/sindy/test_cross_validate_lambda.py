"""Checks for cross_validate_lambda.py against cross_validate_lambda.m."""

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from extensions.sindy.cross_validate_lambda import cross_validate_lambda
from extensions.sindy.stlsq import stlsq


def test_selects_working_lambda_with_noise_and_protected_term():
    rng = np.random.default_rng(0)
    N, n_terms, n_states = 400, 6, 1
    Theta = rng.standard_normal((N, n_terms))

    Xi_true = np.zeros((n_terms, n_states))
    Xi_true[0, 0] = 2.5    # real, unprotected -- should survive on its own merit
    Xi_true[2, 0] = 0.02   # tiny, but PROTECTED -> guaranteed inclusion

    noise = 0.05 * rng.standard_normal((N, n_states))
    dX = Theta @ Xi_true + noise

    protected_idx = np.array([False, False, True, False, False, False])
    cfg = {"n_lambda": 15, "max_iter": 30, "tol": 1e-6}

    lambda_opt = cross_validate_lambda(Theta, dX, cfg, protected_idx=protected_idx,
                                          random_state=0)

    assert np.isfinite(lambda_opt) and lambda_opt > 0

    Xi = stlsq(Theta, dX, lambda_opt, cfg["max_iter"], cfg["tol"], False,
                protected_idx=protected_idx)
    assert Xi[0, 0] != 0.0   # real signal recovered
    assert Xi[2, 0] != 0.0   # protected term always present
    assert Xi[1, 0] == 0.0   # no real signal -- should be pruned
    assert Xi[3, 0] == 0.0
    assert Xi[4, 0] == 0.0
    assert Xi[5, 0] == 0.0


def test_raises_when_every_fold_is_degenerate():
    # dX identically zero -> every fold's fit is exactly zero everywhere
    # (no protected terms to force any survivor), so every lambda in the
    # grid produces a degenerate model -> must raise, not silently
    # return a meaningless lambda.
    rng = np.random.default_rng(1)
    N, n_terms, n_states = 100, 5, 1
    Theta = rng.standard_normal((N, n_terms))
    dX = np.zeros((N, n_states))

    cfg = {"n_lambda": 10, "max_iter": 20, "tol": 1e-6}

    try:
        cross_validate_lambda(Theta, dX, cfg, random_state=0)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "degenerate" in str(e).lower()


def test_composes_with_build_sindy_library_and_stlsq_at_katies_real_dims():
    n_states, n_actions, n_sessions, n_patients = 6, 13, 12, 20
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    lib_cfg = {"poly_order": 2, "include_cross": True, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, lib_cfg)

    cv_cfg = {"n_lambda": 8, "max_iter": 20, "tol": 1e-4,
               "max_terms": 22, "min_terms": 3}

    lambda_opt = cross_validate_lambda(Theta, dX, cv_cfg,
                                          protected_idx=protected_idx, random_state=0)

    assert np.isfinite(lambda_opt) and lambda_opt > 0

    Xi = stlsq(Theta, dX, lambda_opt, cv_cfg["max_iter"], cv_cfg["tol"], False,
                cv_cfg["max_terms"], protected_idx, cv_cfg["min_terms"])
    assert Xi.shape == (Theta.shape[1], n_states)
    assert np.all(np.isfinite(Xi))
    protected_rows = Xi[protected_idx, :]
    assert np.all(np.count_nonzero(protected_rows, axis=0) == protected_rows.shape[0])


if __name__ == "__main__":
    tests = [test_selects_working_lambda_with_noise_and_protected_term,
              test_raises_when_every_fold_is_degenerate,
              test_composes_with_build_sindy_library_and_stlsq_at_katies_real_dims]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
