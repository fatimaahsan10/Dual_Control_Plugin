"""Checks for validate_sindy_model.py against validate_sindy_model.m."""

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from applications.sci_rehab.continuous_state_dynamics import build_sindy_row
from extensions.sindy.cross_validate_lambda import cross_validate_lambda
from extensions.sindy.prune_sindy_bic import prune_sindy_bic
from extensions.sindy.stlsq import stlsq
from extensions.sindy.validate_sindy_model import validate_sindy_model

N_STATES = 6
N_ACTIONS = 13
LIB_CFG = {"poly_order": 1, "include_cross": False, "include_trig": False}
N_TERMS = 1 + (N_STATES + N_ACTIONS)  # bias + linear only


def simulate_true_trajectory(x0, A_p, Xi, cfg):
    """Independent re-implementation of the Euler rollout, used to
    construct a self-consistent X_val whose ground truth is KNOWN to
    follow Xi exactly -- so validate_sindy_model.py's own rollout can be
    checked against it without circularity (build_sindy_row itself is
    already covered by test_continuous_state_dynamics.py)."""
    n_sessions = A_p.shape[1] + 1
    n_states = x0.shape[0]
    X = np.zeros((n_states, n_sessions))
    X[:, 0] = x0
    x = x0.copy()
    for s in range(n_sessions - 1):
        theta_row = build_sindy_row(x, A_p[:, s], cfg)
        dx = theta_row @ Xi
        x = np.clip(x + dx, 0.0, 1.0)
        X[:, s + 1] = x
    return X


def make_linear_xi():
    Xi = np.zeros((N_TERMS, N_STATES))
    u1_term = 1 + N_STATES + 0  # linear coefficient for action a1
    u2_term = 1 + N_STATES + 1  # linear coefficient for action a2
    Xi[u1_term, 0] = 0.10   # dSCIM/dt = 0.10 * u1
    Xi[u2_term, 1] = 0.05   # dBBS/dt  = 0.05 * u2
    return Xi


def test_perfect_model_gives_zero_error():
    rng = np.random.default_rng(0)
    Xi = make_linear_xi()
    n_sessions, n_val = 8, 3

    X_val = np.zeros((N_STATES, n_sessions, n_val))
    A_val = np.zeros((N_ACTIONS, n_sessions, n_val))

    for p in range(n_val):
        x0 = np.array([0.1, 0.1, 0.5, 0.4, 0.3, 0.6])
        A_p = rng.uniform(0, 1, size=(N_ACTIONS, n_sessions - 1))
        A_val[:, :-1, p] = A_p
        X_val[:, :, p] = simulate_true_trajectory(x0, A_p, Xi, LIB_CFG)

    nmae, rmse, sim_trajectories = validate_sindy_model(
        Xi, X_val, A_val, Y_val=None, lib_labels=None, cfg=LIB_CFG)

    assert nmae < 1e-10
    assert rmse < 1e-10
    assert np.allclose(sim_trajectories["X_sim"], X_val, atol=1e-10)
    assert sim_trajectories["state_idx"] == [0, 1]
    assert sim_trajectories["n_val"] == n_val
    assert sim_trajectories["n_sessions"] == n_sessions


def test_error_metrics_match_hand_computation():
    # Xi = 0 everywhere -> the model predicts NO change at all, so
    # X_sim stays frozen at the initial state for every session --
    # makes NMAE/RMSE fully hand-computable from X_val alone.
    Xi = np.zeros((N_TERMS, N_STATES))
    n_sessions = 3

    X_val = np.zeros((N_STATES, n_sessions, 1))
    X_val[0, :, 0] = [0.2, 0.5, 0.9]   # SCIM
    X_val[1, :, 0] = [0.1, 0.3, 0.5]   # BBS
    X_val[2:, :, 0] = 0.5              # static states (irrelevant to scoring)
    A_val = np.zeros((N_ACTIONS, n_sessions, 1))

    nmae, rmse, sim_trajectories = validate_sindy_model(
        Xi, X_val, A_val, Y_val=None, lib_labels=None, cfg=LIB_CFG)

    err = np.array([0, 0.3, 0.7, 0, 0.2, 0.4])  # [SCIM row, BBS row] errors
    expected_rmse = np.sqrt(np.mean(err ** 2))
    dyn_range = 0.9 - 0.1
    expected_nmae = np.mean(np.abs(err)) / dyn_range

    assert np.isclose(rmse, expected_rmse)
    assert np.isclose(nmae, expected_nmae)


def test_scoring_ignores_static_state_mismatches():
    Xi = make_linear_xi()  # zero coefficients for static states 2-5
    n_sessions = 4

    x0 = np.array([0.1, 0.1, 0.5, 0.4, 0.3, 0.6])
    A_p = np.zeros((N_ACTIONS, n_sessions - 1))
    A_p[0, :] = 0.3
    A_p[1, :] = 0.2

    X_val = simulate_true_trajectory(x0, A_p, Xi, LIB_CFG)[:, :, None]
    # engineer a static-state "drift" in the ground truth that Xi's
    # (zero) static-state coefficients can never reproduce
    X_val[2, :, 0] = [0.5, 0.6, 0.7, 0.8]
    A_val = np.concatenate([A_p, A_p[:, -1:]], axis=1)[:, :, None]

    nmae, rmse, _ = validate_sindy_model(
        Xi, X_val, A_val, Y_val=None, lib_labels=None, cfg=LIB_CFG)

    assert nmae < 1e-10  # dynamic states (0,1) still match perfectly
    assert rmse < 1e-10


def test_composes_with_full_pipeline_at_katies_real_dims():
    n_states, n_actions, n_sessions, n_patients = 6, 13, 12, 30
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(n_states, n_sessions, n_patients))
    A = rng.uniform(0, 1, size=(n_actions, n_sessions, n_patients))

    val_idx = np.arange(24, 30)
    train_idx = np.arange(0, 24)

    lib_cfg = {"poly_order": 2, "include_cross": True, "include_trig": False}
    Theta, dX, labels, protected_idx = build_sindy_library(
        X[:, :, train_idx], A[:, :, train_idx], lib_cfg)

    cv_cfg = {"n_lambda": 8, "max_iter": 20, "tol": 1e-4,
               "max_terms": 22, "min_terms": 3}
    lambda_opt = cross_validate_lambda(Theta, dX, cv_cfg,
                                          protected_idx=protected_idx, random_state=0)
    Xi = stlsq(Theta, dX, lambda_opt, cv_cfg["max_iter"], cv_cfg["tol"], False,
                cv_cfg["max_terms"], protected_idx, cv_cfg["min_terms"])
    state_names = ["SCIM", "BBS", "AIS", "Age", "DPI", "Caregiver"]
    Xi, _ = prune_sindy_bic(Xi, Theta, dX, labels, state_names, False, protected_idx)
    Xi[:, 2:] = 0.0  # main_sindy_pipeline.m's "zero static state equations" step

    nmae, rmse, sim_trajectories = validate_sindy_model(
        Xi, X[:, :, val_idx], A[:, :, val_idx], Y_val=None, lib_labels=labels,
        cfg=lib_cfg)

    assert np.isfinite(nmae) and nmae >= 0
    assert np.isfinite(rmse) and rmse >= 0
    assert sim_trajectories["X_sim"].shape == (n_states, n_sessions, len(val_idx))
    assert np.all((sim_trajectories["X_sim"] >= 0) & (sim_trajectories["X_sim"] <= 1))


if __name__ == "__main__":
    tests = [test_perfect_model_gives_zero_error,
              test_error_metrics_match_hand_computation,
              test_scoring_ignores_static_state_mismatches,
              test_composes_with_full_pipeline_at_katies_real_dims]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
