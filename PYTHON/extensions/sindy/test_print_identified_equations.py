"""Checks for print_identified_equations.py against print_identified_equations.m."""

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from extensions.sindy.cross_validate_lambda import cross_validate_lambda
from extensions.sindy.print_identified_equations import print_identified_equations
from extensions.sindy.prune_sindy_bic import prune_sindy_bic
from extensions.sindy.stlsq import stlsq

STATE_NAMES = ["SCIM", "BBS", "AIS", "Age", "DPI", "Caregiver"]


def test_equation_prints_correctly_and_strips_scale_tags():
    n_terms, n_states = 4, 2
    Xi = np.zeros((n_terms, n_states))
    Xi[0, 0] = 0.5
    Xi[1, 0] = -0.25
    lib_labels = ["1 [s=1.000]", "a7 [s=0.500]", "x1 [s=0.300]", "x2^2 [s=0.100]"]

    report = print_identified_equations(Xi, lib_labels, STATE_NAMES[:2], cfg={})

    assert "d(SCIM)/dt = +0.5000 * 1  -0.2500 * a7" in report
    assert "[s=" not in report  # scale tags fully stripped
    assert "d(BBS)/dt = 0  (all coefficients zeroed)" in report


def test_clinical_interpretation_translates_top_two_terms_only():
    n_terms, n_states = 4, 1
    Xi = np.zeros((n_terms, n_states))
    Xi[0, 0] = 0.30   # 3rd-largest -> should NOT appear in interpretation
    Xi[1, 0] = 0.80    # largest -> "increases"
    Xi[2, 0] = -0.50   # 2nd-largest -> "decreases"
    lib_labels = ["x2^2 [s=1.0]", "a7 [s=1.0]", "x1 [s=1.0]", "a13 [s=1.0]"]

    report = print_identified_equations(Xi, lib_labels, ["SCIM"], cfg={})

    assert "U7 increases SCIM" in report
    assert "SCIM decreases SCIM" in report  # x1 -> SCIM
    assert "x2 squared" not in report  # 3rd-largest term excluded from interpretation
    # but the 3rd-largest term IS still shown in the equation itself
    assert "x2^2" in report


def test_a_dollar_1_does_not_get_confused_with_a10_through_a13():
    n_terms, n_states = 2, 1
    Xi = np.array([[0.9], [0.8]])
    lib_labels = ["a1 [s=1.0]", "a10 [s=1.0]"]

    report = print_identified_equations(Xi, lib_labels, ["SCIM"], cfg={})

    assert "U1 increases SCIM" in report
    assert "U10 increases SCIM" in report
    assert "U100" not in report  # would indicate a1 wrongly matched inside a10


def test_sparsity_summary_uses_exact_nonzero_not_print_threshold():
    # 1e-9 is below the 1e-8 print threshold (excluded from the printed
    # equation) but is NOT exactly zero (must still count in the
    # summary's np.count_nonzero) -- the two checks are deliberately
    # different, matching the MATLAB source.
    n_terms, n_states = 3, 2
    Xi = np.zeros((n_terms, n_states))
    Xi[0, 0] = 1e-9
    Xi[1, 1] = 0.5
    lib_labels = ["1 [s=1.0]", "a1 [s=1.0]", "x1 [s=1.0]"]

    report = print_identified_equations(Xi, lib_labels, STATE_NAMES[:2], cfg={})

    assert "d(SCIM)/dt = 0  (all coefficients zeroed)" in report  # 1e-9 excluded here
    assert "Non-zero coefficients (dynamic states): 2 / 6" in report  # but counted here


def test_composes_with_full_pipeline_at_katies_real_dims():
    n_states, n_actions, n_sessions, n_patients = 6, 13, 12, 20
    rng = np.random.default_rng(0)
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
    Xi, _ = prune_sindy_bic(Xi, Theta, dX, labels, STATE_NAMES, False, protected_idx)
    Xi[:, 2:] = 0.0

    report = print_identified_equations(Xi, labels, STATE_NAMES, cfg={})

    assert "SINDy Identified Equations" in report
    assert "d(SCIM)/dt" in report and "d(BBS)/dt" in report
    expected_nnz = np.count_nonzero(Xi[:, :2])
    assert f"Non-zero coefficients (dynamic states): {expected_nnz} / {Xi[:, :2].size}" in report


if __name__ == "__main__":
    tests = [test_equation_prints_correctly_and_strips_scale_tags,
              test_clinical_interpretation_translates_top_two_terms_only,
              test_a_dollar_1_does_not_get_confused_with_a10_through_a13,
              test_sparsity_summary_uses_exact_nonzero_not_print_threshold,
              test_composes_with_full_pipeline_at_katies_real_dims]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
