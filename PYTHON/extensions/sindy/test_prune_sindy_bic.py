"""Checks for prune_sindy_bic.py against prune_sindy_bic.m."""

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from extensions.sindy.cross_validate_lambda import cross_validate_lambda
from extensions.sindy.prune_sindy_bic import prune_sindy_bic
from extensions.sindy.stlsq import stlsq


def test_prunes_marginal_term_kept_by_permissive_stlsq():
    rng = np.random.default_rng(0)
    N, n_terms = 300, 5
    Theta = rng.standard_normal((N, n_terms))
    true_signal = 2.0 * Theta[:, 0] + 1.5 * Theta[:, 1]
    dX = (true_signal + 0.1 * rng.standard_normal(N)).reshape(N, 1)

    # permissive lambda -> STLSQ keeps some noise-driven terms too
    Xi_before = stlsq(Theta, dX, lam=0.002, max_iter=50, tol=1e-8, verbose=False)
    n_active_before = np.count_nonzero(Xi_before[:, 0])
    assert n_active_before > 2, "test setup expects STLSQ to over-select"

    lib_labels = [f"term{j} [s=1.000]" for j in range(n_terms)]
    Xi_pruned, bic_history = prune_sindy_bic(
        Xi_before, Theta, dX, lib_labels, state_names=["SCIM"], verbose=False)

    n_active_after = np.count_nonzero(Xi_pruned[:, 0])
    assert n_active_after < n_active_before
    assert Xi_pruned[0, 0] != 0.0  # real terms survive
    assert Xi_pruned[1, 0] != 0.0

    assert len(bic_history) == 1
    trace = bic_history[0]
    assert trace["state"] == "SCIM"
    assert trace["nnz_vals"][0] == n_active_before
    assert trace["nnz_vals"][-1] == n_active_after
    assert np.all(np.diff(trace["nnz_vals"]) == -1)  # drops exactly one per accepted step


def test_protected_term_never_dropped_by_bic():
    rng = np.random.default_rng(1)
    N, n_terms = 300, 4
    Theta = rng.standard_normal((N, n_terms))
    # dX driven only by term0 -- terms 1 and 2 are both spurious/negligible
    dX = (3.0 * Theta[:, 0] + 0.05 * rng.standard_normal(N)).reshape(N, 1)

    Xi_before = np.zeros((n_terms, 1))
    Xi_before[0, 0] = 3.0
    Xi_before[1, 0] = 0.05   # spurious, unprotected -> should be pruned
    Xi_before[2, 0] = 0.02   # spurious, PROTECTED -> must survive

    protected_idx = np.array([False, False, True, False])
    lib_labels = [f"term{j} [s=1.000]" for j in range(n_terms)]

    Xi_pruned, bic_history = prune_sindy_bic(
        Xi_before, Theta, dX, lib_labels, state_names=["SCIM"], verbose=False,
        protected_idx=protected_idx)

    assert Xi_pruned[0, 0] != 0.0   # real term survives
    assert Xi_pruned[1, 0] == 0.0   # spurious unprotected term pruned
    assert Xi_pruned[2, 0] != 0.0   # protected term survives despite being spurious


def test_only_dynamic_states_are_touched():
    rng = np.random.default_rng(2)
    N, n_terms, n_states = 200, 6, 6
    Theta = rng.standard_normal((N, n_terms))
    dX = rng.standard_normal((N, n_states)) * 0.1

    Xi_before = np.zeros((n_terms, n_states))
    Xi_before[0, 0] = 2.0
    Xi_before[1, 0] = 0.01  # SCIM: real + spurious -> should get pruned
    Xi_before[0, 1] = 2.0
    Xi_before[1, 1] = 0.01  # BBS: same pattern
    # states 2-5 (static: AIS/Age/DPI/Caregiver) get some arbitrary
    # nonzero pattern that should be left COMPLETELY untouched
    Xi_before[2, 2] = 5.0
    Xi_before[3, 3] = -1.0
    Xi_before[0, 4] = 7.0
    Xi_before[4, 5] = 0.5

    lib_labels = [f"term{j} [s=1.000]" for j in range(n_terms)]
    state_names = ["SCIM", "BBS", "AIS", "Age", "DPI", "Caregiver"]

    Xi_pruned, bic_history = prune_sindy_bic(
        Xi_before, Theta, dX, lib_labels, state_names, verbose=False)

    assert len(bic_history) == 2  # only SCIM and BBS processed
    assert {h["state"] for h in bic_history} == {"SCIM", "BBS"}
    # static states (columns 2-5) must be byte-for-byte untouched
    assert np.array_equal(Xi_pruned[:, 2:], Xi_before[:, 2:])


def test_composes_with_full_pipeline_at_katies_real_dims():
    n_states, n_actions, n_sessions, n_patients = 6, 13, 12, 20
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    lib_cfg = {"poly_order": 2, "include_cross": True, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, lib_cfg)

    cv_cfg = {"n_lambda": 8, "max_iter": 20, "tol": 1e-4,
               "max_terms": 22, "min_terms": 3}
    lambda_opt = cross_validate_lambda(Theta, dX, cv_cfg,
                                          protected_idx=protected_idx, random_state=0)
    Xi = stlsq(Theta, dX, lambda_opt, cv_cfg["max_iter"], cv_cfg["tol"], False,
                cv_cfg["max_terms"], protected_idx, cv_cfg["min_terms"])

    state_names = ["SCIM", "BBS", "AIS", "Age", "DPI", "Caregiver"]
    Xi_pruned, bic_history = prune_sindy_bic(
        Xi, Theta, dX, labels, state_names, verbose=False, protected_idx=protected_idx)

    assert Xi_pruned.shape == Xi.shape
    assert np.all(np.isfinite(Xi_pruned))
    assert len(bic_history) == 2
    # pruning must never increase nnz, and never touch static states
    for s in range(2):
        assert np.count_nonzero(Xi_pruned[:, s]) <= np.count_nonzero(Xi[:, s])
    assert np.array_equal(Xi_pruned[:, 2:], Xi[:, 2:])
    # protected action terms must survive pruning in the two dynamic states
    for s in range(2):
        assert np.all(Xi_pruned[protected_idx, s] != 0)


if __name__ == "__main__":
    tests = [test_prunes_marginal_term_kept_by_permissive_stlsq,
              test_protected_term_never_dropped_by_bic,
              test_only_dynamic_states_are_touched,
              test_composes_with_full_pipeline_at_katies_real_dims]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
