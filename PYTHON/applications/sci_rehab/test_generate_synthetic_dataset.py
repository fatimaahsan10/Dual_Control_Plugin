"""Checks for generate_synthetic_dataset.py against generate_synthetic_dataset.m."""

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from applications.sci_rehab.define_persona import Persona, define_persona
from applications.sci_rehab.generate_synthetic_dataset import generate_synthetic_dataset
from applications.sci_rehab.who_therapies import who_therapies


def _neutral_persona(therapy_efficacy=None):
    """A Persona with no attenuation/amplification (motivation=1,
    pain_sensitivity=0, fatigue_tol=1 -> load_atten always 1) so
    u_eff == u * therapy_efficacy exactly -- isolates the core state-
    transition formula from persona-modulation complexity for
    hand-verification tests."""
    return Persona(
        ais_probs=np.array([0.25, 0.25, 0.25, 0.25]), age_range=(30, 50),
        dpi_range=(10, 40), caregiver_alpha=2, caregiver_beta=2,
        motivation=1.0, fatigue_tol=1.0, pain_sensitivity=0.0, comorbidity=0.0,
        therapy_efficacy=(therapy_efficacy if therapy_efficacy is not None
                              else np.ones(13)),
        w_scim=2.0, w_bbs=1.0, w_resource=1e-2,
        w_terminal_scim=50.0, w_terminal_bbs=25.0,
        name="Neutral", description="test-only neutral persona")


def test_shapes_default_persona():
    cfg = {"n_patients": 5, "n_sessions": 6, "noise_std": 0.01}
    ds = generate_synthetic_dataset(cfg, rng=np.random.default_rng(0))

    assert ds.X.shape == (6, 6, 5)
    assert ds.A.shape == (13, 6, 5)
    assert ds.Y.shape == (1, 6, 5)
    assert ds.n_states == 6 and ds.n_actions == 13 and ds.n_patients == 5
    assert ds.persona.name == "Default"


def test_reproducible_with_same_rng_seed():
    cfg = {"n_patients": 10, "n_sessions": 8, "noise_std": 0.02,
            "persona": define_persona("young_athlete")}
    ds1 = generate_synthetic_dataset(cfg, rng=np.random.default_rng(42))
    ds2 = generate_synthetic_dataset(cfg, rng=np.random.default_rng(42))

    assert np.array_equal(ds1.X, ds2.X)
    assert np.array_equal(ds1.A, ds2.A)
    assert np.array_equal(ds1.swat_ceiling, ds2.swat_ceiling)


def test_sparse_activation_3_to_5_therapies_per_session():
    cfg = {"n_patients": 8, "n_sessions": 10, "noise_std": 0.0}
    ds = generate_synthetic_dataset(cfg, rng=np.random.default_rng(1))

    n_active = np.count_nonzero(ds.A[:, :-1, :], axis=0)  # exclude terminal (repeated) column
    assert np.all(n_active >= 3) and np.all(n_active <= 5)


def test_terminal_session_action_repeats_previous():
    cfg = {"n_patients": 6, "n_sessions": 9, "noise_std": 0.0}
    ds = generate_synthetic_dataset(cfg, rng=np.random.default_rng(2))

    assert np.array_equal(ds.A[:, -1, :], ds.A[:, -2, :])


def test_static_states_exactly_constant_when_noise_free():
    cfg = {"n_patients": 4, "n_sessions": 7, "noise_std": 0.0}
    ds = generate_synthetic_dataset(cfg, rng=np.random.default_rng(3))

    for p in range(4):
        for s in range(7):
            assert np.array_equal(ds.X[2:, s, p], ds.X[2:, 0, p])


def test_dynamic_states_and_ceiling_stay_in_bounds():
    cfg = {"n_patients": 15, "n_sessions": 12, "noise_std": 0.05,
            "persona": define_persona("young_athlete")}
    ds = generate_synthetic_dataset(cfg, rng=np.random.default_rng(4))

    assert np.all(ds.X[:2, :, :] >= 0.0) and np.all(ds.X[:2, :, :] <= 1.0)
    assert np.all(ds.swat_ceiling >= 0.1) and np.all(ds.swat_ceiling <= 1.0)


def test_one_step_dynamics_matches_hand_formula():
    # noise_std=0 + a neutral persona (u_eff == u exactly) makes the
    # first session's state update fully hand-verifiable from the
    # recorded X/A alone, independent of the module's own internals.
    # n_sessions=4 (not the minimal 2) only so the module's internal
    # Xi_true label-generation call to build_sindy_library.py sees at
    # least one interior session -- avoids a harmless but noisy ddof=1
    # RuntimeWarning on a degenerate (zero-row) slice; this test only
    # ever checks the session 0->1 transition regardless.
    persona = _neutral_persona()
    cfg = {"n_patients": 1, "n_sessions": 4, "noise_std": 0.0, "persona": persona}
    ds = generate_synthetic_dataset(cfg, rng=np.random.default_rng(5))

    x0 = ds.X[:, 0, 0]
    x1_scim, x1_bbs, x3, x4, x5, x6 = x0
    u = ds.A[:, 0, 0]  # prescribed dose == effective dose for a neutral persona

    T = who_therapies()
    ceil_p = ds.swat_ceiling[0]

    head1 = np.clip(1 - x1_scim / max(ceil_p, 0.05), 0, 1)
    head2 = np.clip(1 - x1_bbs / max(ceil_p, 0.05), 0, 1)
    contrib_scim = (T.G[0, :] @ u) * (1 + 0.30 * x3 + 0.15 * x6)
    contrib_bbs = (T.G[1, :] @ u) * (1 + 0.15 * x6)
    soft1 = 0.55 + 0.45 * head1
    soft2 = 0.55 + 0.45 * head2

    expected_dx1 = (-0.05 * x1_scim + 0.30 * x1_scim * x3) * head1 + contrib_scim * soft1 * 0.20
    expected_dx2 = (-0.04 * x1_bbs + 0.22 * x1_bbs * x3) * head2 + contrib_bbs * soft2 * 0.20

    x_next = ds.X[:, 1, 0]
    assert np.isclose(x_next[0], np.clip(x1_scim + expected_dx1, 0, 1), atol=1e-8)
    assert np.isclose(x_next[1], np.clip(x1_bbs + expected_dx2, 0, 1), atol=1e-8)


def test_xi_true_structure_and_zero_efficacy_zeroes_its_row():
    n_actions = 13
    eff = np.ones(n_actions)
    eff[0] = 0.0  # therapy U1 has zero efficacy for this persona
    persona = _neutral_persona(therapy_efficacy=eff)
    cfg = {"n_patients": 3, "n_sessions": 6, "noise_std": 0.0, "persona": persona}
    ds = generate_synthetic_dataset(cfg, rng=np.random.default_rng(6))

    assert ds.Xi_true.shape[1] == ds.n_states
    assert np.array_equal(ds.Xi_true[:, 2:], np.zeros((ds.Xi_true.shape[0], 4)))
    # exactly 4 base state terms + 2 per action term, minus the 2 zeroed
    # by U1's zero efficacy = 4 + 2*13 - 2 = 28
    assert np.count_nonzero(ds.Xi_true) == 28

    lib_cfg = {"poly_order": 2, "include_cross": True, "include_trig": False}
    _, _, labels, _ = build_sindy_library(ds.X[:, :, :2], ds.A[:, :, :2], lib_cfg)
    clean = [lbl.split(" [s=")[0] for lbl in labels]
    a1_row = clean.index("a1")
    assert np.allclose(ds.Xi_true[a1_row, :2], [0.0, 0.0])


if __name__ == "__main__":
    tests = [test_shapes_default_persona,
              test_reproducible_with_same_rng_seed,
              test_sparse_activation_3_to_5_therapies_per_session,
              test_terminal_session_action_repeats_previous,
              test_static_states_exactly_constant_when_noise_free,
              test_dynamic_states_and_ceiling_stay_in_bounds,
              test_one_step_dynamics_matches_hand_formula,
              test_xi_true_structure_and_zero_efficacy_zeroes_its_row]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
