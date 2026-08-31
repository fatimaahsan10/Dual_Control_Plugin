"""Checks for define_persona.py against define_persona.m."""

import warnings

import numpy as np

from applications.sci_rehab.define_persona import _DEFAULT_PERSONA, _PERSONAS, define_persona, persona_weights
from applications.sci_rehab.sindy_model_constants import PersonaWeights, SindyModelConstants
from applications.sci_rehab.l_cost import l_cost

ALL_PERSONA_KEYS = list(_PERSONAS.keys()) + ["default"]


def test_known_persona_returns_correct_data():
    p = define_persona("high_motivator")
    assert p.name == "High Motivator"
    assert np.isclose(p.w_scim, 2.5)
    assert np.isclose(p.w_resource, 5e-3)
    assert np.allclose(p.therapy_efficacy,
                          [1.1, 1.0, 1.3, 1.2, 0.9, 1.0, 1.2, 1.3, 0.9, 1.0, 1.0, 1.2, 1.3])
    assert np.isclose(np.sum(p.ais_probs), 1.0)


def test_name_matching_is_case_insensitive_and_trimmed():
    p1 = define_persona("high_motivator")
    p2 = define_persona("  High_Motivator  ")
    p3 = define_persona("HIGH_MOTIVATOR")
    assert p1.name == p2.name == p3.name == "High Motivator"


def test_all_18_personas_valid_via_public_api():
    for key in ALL_PERSONA_KEYS:
        p = define_persona(key)
        assert np.isclose(np.sum(p.ais_probs), 1.0), key
        assert len(p.therapy_efficacy) == 13, key
        assert p.name != ""


def test_unknown_name_warns_and_falls_back_to_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = define_persona("not_a_real_persona")

    assert p.name == "Default"
    assert any("unknown persona" in str(w.message) for w in caught)


def test_explicit_default_returns_default_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = define_persona("default")
        p2 = define_persona("Default")

    assert p.name == "Default"
    assert p2.name == "Default"
    assert len(caught) == 0  # no warning for the literal string "default"


def test_persona_weights_bridges_to_l_cost():
    p = define_persona("frail_elder")
    weights = persona_weights(p)

    assert isinstance(weights, PersonaWeights)
    assert weights.w_scim == p.w_scim
    assert weights.w_bbs == p.w_bbs
    assert weights.w_resource == p.w_resource
    assert weights.w_terminal_scim == p.w_terminal_scim
    assert weights.w_terminal_bbs == p.w_terminal_bbs

    # end-to-end: plug straight into SindyModelConstants / l_cost.py
    constants = SindyModelConstants(Xi=np.zeros((1, 6)), lib_cfg={},
                                       swat_ceiling=0.8, xi_idx=np.zeros((0, 2)),
                                       persona=weights)
    xa = np.array([[0.3], [0.5], [0.5], [0.4], [0.3], [0.6]])
    u = np.full((13, 1), 0.2)
    cost = l_cost(xa, u, tracking_trajectory=None, u_lims=None, u_lim_method=1,
                    constants=constants)
    gap_scim, gap_bbs = 0.8 - 0.3, 0.8 - 0.5
    expected = (p.w_scim * gap_scim ** 2 + p.w_bbs * gap_bbs ** 2
                 + p.w_resource * 13 * 0.2 ** 2 + 1e-6 * 13 * 0.2 ** 2)
    assert np.isclose(cost[0], expected)


def test_sgba_personas_are_concentrated_on_one_ais_grade():
    sgba_keys = ["p1_maya", "sophia", "p3_riley", "jean_guy", "p5_daniel", "joan",
                  "kenny", "p8_marcus", "p9_aaliyah", "p10_pierre", "p11_niran",
                  "p12_lucia"]
    archetype_keys = ["high_motivator", "frail_elder", "young_athlete",
                        "low_support", "pain_sensitive"]

    for key in sgba_keys:
        p = define_persona(key)
        assert np.max(p.ais_probs) >= 0.9, f"{key} should be concentrated on one grade"

    for key in archetype_keys:
        p = define_persona(key)
        assert np.max(p.ais_probs) < 0.9, f"{key} should be spread across a population"


if __name__ == "__main__":
    tests = [test_known_persona_returns_correct_data,
              test_name_matching_is_case_insensitive_and_trimmed,
              test_all_18_personas_valid_via_public_api,
              test_unknown_name_warns_and_falls_back_to_default,
              test_explicit_default_returns_default_without_warning,
              test_persona_weights_bridges_to_l_cost,
              test_sgba_personas_are_concentrated_on_one_ais_grade]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
