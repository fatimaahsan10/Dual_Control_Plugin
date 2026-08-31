"""Checks for build_sindy_library.py against build_sindy_library.m."""

import math
import re

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library


def test_finite_differences_recover_known_linear_trajectory():
    # Each state trajectory is exactly linear in session index, so the
    # true derivative is a known constant (`slope`) at every interior
    # point regardless of whether a forward/central/backward stencil is
    # used -- a clean way to check the finite-difference machinery AND
    # that only interior sessions get stacked into rows.
    n_states, n_actions, n_sessions, n_patients = 6, 13, 5, 3
    rng = np.random.default_rng(0)

    slope = rng.uniform(0.05, 0.2, size=(n_states, 1, n_patients))
    x0 = rng.uniform(0.1, 0.5, size=(n_states, 1, n_patients))
    sessions = np.arange(n_sessions).reshape(1, n_sessions, 1)
    X = x0 + slope * sessions  # (n_states, n_sessions, n_patients)
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    cfg = {"poly_order": 1, "include_cross": False, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, cfg)

    n_interior = n_sessions - 2
    assert dX.shape == (n_patients * n_interior, n_states)
    # every row's derivative must equal that patient's constant slope
    expected_rows = np.repeat(slope[:, 0, :].T, n_interior, axis=0)
    assert np.allclose(dX, expected_rows, atol=1e-10)


def test_theta_columns_match_hand_computation_small_case():
    # n_states=2, n_actions=1 -> n_vars=3, two interior sessions -> N=2
    # rows (need >=2 so the N-1-normalised column std used for the
    # "[s=...]" label tag isn't a degenerate 0/0 NaN), so the first
    # interior session's Theta row can still be checked by hand.
    n_states, n_actions, n_sessions, n_patients = 2, 1, 4, 1
    X = np.zeros((n_states, n_sessions, n_patients))
    X[:, :, 0] = np.array([[0.1, 0.3, 0.6, 0.65], [0.2, 0.25, 0.35, 0.5]])
    A = np.zeros((n_actions, n_sessions, n_patients))
    A[:, :, 0] = np.array([[0.5, 0.7, 0.9, 0.8]])

    cfg = {"poly_order": 3, "include_cross": True, "include_trig": True}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, cfg)

    x1, x2, a1 = 0.3, 0.25, 0.7  # values at the first interior session (s=1)
    expected = np.array([
        1.0, x1, x2, a1,                     # bias, linear
        x1 ** 2, x2 ** 2, a1 ** 2,           # squared
        x1 * x2, x1 * a1, x2 * a1,           # cross (pairs i<j over [x1,x2,a1])
        x1 ** 3, x2 ** 3,                    # cubic (first 2 states only)
        np.sin(x1), np.sin(x2), np.sin(a1),  # sin
        np.cos(x1), np.cos(x2), np.cos(a1),  # cos
    ])

    assert Theta.shape == (2, 18)
    assert len(labels) == 18
    assert np.allclose(Theta[0, :], expected, atol=1e-10)

    raw_labels = [re.sub(r"\s*\[s=[\d.]+\]", "", lbl) for lbl in labels]
    assert raw_labels == ["1", "x1", "x2", "a1", "x1^2", "x2^2", "a1^2",
                            "x1*x2", "x1*a1", "x2*a1", "x1^3", "x2^3",
                            "sin(x1)", "sin(x2)", "sin(a1)",
                            "cos(x1)", "cos(x2)", "cos(a1)"]

    # expected dX at the first interior session: central difference
    expected_dx1 = (0.6 - 0.1) / 2
    expected_dx2 = (0.35 - 0.2) / 2
    assert np.allclose(dX[0, :], [expected_dx1, expected_dx2])


def test_column_scale_tag_uses_matlab_ddof1_convention():
    n_states, n_actions, n_sessions, n_patients = 2, 1, 4, 2
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    cfg = {"poly_order": 1, "include_cross": False, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, cfg)

    x1_col_idx = 1  # labels[0]='1', labels[1]='x1 [s=...]'
    expected_std = np.std(Theta[:, x1_col_idx], ddof=1)
    m = re.search(r"\[s=([\d.]+)\]", labels[x1_col_idx])
    assert np.isclose(float(m.group(1)), expected_std, atol=1e-3)

    # ddof=1 must differ from numpy's population-std default (ddof=0)
    # for this to be a meaningful check
    assert not np.isclose(expected_std, np.std(Theta[:, x1_col_idx], ddof=0))


def test_protected_idx_marks_default_action_terms_and_user_terms():
    n_states, n_actions, n_sessions, n_patients = 2, 3, 4, 2
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    cfg = {"poly_order": 2, "include_cross": True, "include_trig": False,
            "protected_terms": ["x1*x2"]}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, cfg)

    raw_labels = [re.sub(r"\s*\[s=[\d.]+\]", "", lbl) for lbl in labels]
    expected_protected = {"a1", "a2", "a3", "x1*x2"}
    for lbl, is_protected in zip(raw_labels, protected_idx):
        assert is_protected == (lbl in expected_protected), lbl
    assert protected_idx.sum() == len(expected_protected)


def test_shapes_and_term_count_at_katies_real_dims():
    n_states, n_actions, n_sessions, n_patients = 6, 13, 12, 20
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    cfg = {"poly_order": 3, "include_cross": True, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(X, A, cfg)

    n_vars = n_states + n_actions
    expected_n_terms = (1 + n_vars + n_vars + math.comb(n_vars, 2) + 2)
    N = n_patients * (n_sessions - 2)

    assert Theta.shape == (N, expected_n_terms)
    assert dX.shape == (N, n_states)
    assert len(labels) == expected_n_terms
    assert protected_idx.shape == (expected_n_terms,)
    assert protected_idx.sum() == n_actions  # default: a1..a13 only


if __name__ == "__main__":
    tests = [test_finite_differences_recover_known_linear_trajectory,
              test_theta_columns_match_hand_computation_small_case,
              test_column_scale_tag_uses_matlab_ddof1_convention,
              test_protected_idx_marks_default_action_terms_and_user_terms,
              test_shapes_and_term_count_at_katies_real_dims]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
