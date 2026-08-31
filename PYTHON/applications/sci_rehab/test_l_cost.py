"""Checks for l_cost.py against l_cost.m, at Katie's dims (nu=13)."""

import numpy as np

from applications.sci_rehab.l_cost import l_cost
from applications.sci_rehab.sindy_model_constants import PersonaWeights, SindyModelConstants

NX = 6
NU = 13


def make_xa(scim, bbs, n_cols):
    xa = np.zeros((NX, n_cols))
    xa[0, :] = scim
    xa[1, :] = bbs
    return xa


def make_constants(swat=0.8):
    weights = PersonaWeights(w_scim=2.5, w_bbs=1.5, w_resource=0.1,
                                w_terminal_scim=60.0, w_terminal_bbs=30.0)
    return SindyModelConstants(Xi=np.zeros((1, NX)), lib_cfg={},
                                  swat_ceiling=swat, xi_idx=np.zeros((0, 2)),
                                  persona=weights)


def test_running_cost_matches_hand_computation():
    constants = make_constants(swat=0.8)
    xa = make_xa(scim=0.3, bbs=0.5, n_cols=1)
    u = np.full((NU, 1), 0.2)

    c = l_cost(xa, u, tracking_trajectory=None, u_lims=None, u_lim_method=1,
                constants=constants)

    gap_scim, gap_bbs = 0.8 - 0.3, 0.8 - 0.5
    expected_lx = 2.5 * gap_scim ** 2 + 1.5 * gap_bbs ** 2
    expected_lu = 0.1 * NU * 0.2 ** 2 + 1e-6 * NU * 0.2 ** 2
    assert np.isclose(c[0], expected_lx + expected_lu)


def test_final_nan_column_gets_terminal_cost_and_zero_resource_cost():
    constants = make_constants(swat=0.8)
    xa = make_xa(scim=0.3, bbs=0.5, n_cols=2)
    u = np.full((NU, 2), 0.2)
    u[:, 1] = np.nan  # final step: NaN-poisoned control

    c = l_cost(xa, u, tracking_trajectory=None, u_lims=None, u_lim_method=1,
                constants=constants)

    gap_scim, gap_bbs = 0.8 - 0.3, 0.8 - 0.5
    lx = 2.5 * gap_scim ** 2 + 1.5 * gap_bbs ** 2
    lf = 60.0 * gap_scim ** 2 + 30.0 * gap_bbs ** 2

    assert np.isclose(c[0], lx + (0.1 * NU * 0.2 ** 2 + 1e-6 * NU * 0.2 ** 2))
    assert np.isclose(c[1], lx + lf)  # resource cost 0 (u zeroed), terminal added


def test_tanh_squashing_applied_to_resource_cost_only_via_u_phys():
    constants = make_constants(swat=0.8)
    xa = make_xa(scim=0.3, bbs=0.5, n_cols=1)
    u = np.full((NU, 1), 5.0)  # far outside bounds
    u_lims = np.tile(np.array([[0.0, 1.0]]), (NU, 1))

    c = l_cost(xa, u, tracking_trajectory=None, u_lims=u_lims, u_lim_method=2,
                constants=constants)

    u_phys = (1.0 / 2 * np.tanh(5.0) + 0.5)  # squash formula for lims=[0,1]
    gap_scim, gap_bbs = 0.8 - 0.3, 0.8 - 0.5
    expected_lx = 2.5 * gap_scim ** 2 + 1.5 * gap_bbs ** 2
    expected_lu = 0.1 * NU * u_phys ** 2 + 1e-6 * NU * 5.0 ** 2  # raw u in the 1e-6 term
    assert np.isclose(c[0], expected_lx + expected_lu)


def test_default_weights_used_when_constants_is_none():
    xa = make_xa(scim=0.3, bbs=0.5, n_cols=1)
    u = np.full((NU, 1), 0.2)

    c = l_cost(xa, u, tracking_trajectory=None, u_lims=None, u_lim_method=1,
                constants=None)

    gap_scim, gap_bbs = 1.0 - 0.3, 1.0 - 0.5  # default swat = 1.0
    expected_lx = 2.0 * gap_scim ** 2 + 1.0 * gap_bbs ** 2  # default w_scim/w_bbs
    expected_lu = 1e-2 * NU * 0.2 ** 2 + 1e-6 * NU * 0.2 ** 2
    assert np.isclose(c[0], expected_lx + expected_lu)


def test_does_not_mutate_callers_u_array():
    constants = make_constants()
    xa = make_xa(scim=0.3, bbs=0.5, n_cols=1)
    u = np.full((NU, 1), np.nan)
    u_copy_before = u.copy()

    l_cost(xa, u, tracking_trajectory=None, u_lims=None, u_lim_method=1,
            constants=constants)

    assert np.array_equal(u, u_copy_before, equal_nan=True)


if __name__ == "__main__":
    tests = [test_running_cost_matches_hand_computation,
              test_final_nan_column_gets_terminal_cost_and_zero_resource_cost,
              test_tanh_squashing_applied_to_resource_cost_only_via_u_phys,
              test_default_weights_used_when_constants_is_none,
              test_does_not_mutate_callers_u_array]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
