"""Checks for measurement.py against Measurement.m."""

import numpy as np

from applications.sci_rehab.measurement import measurement

NX = 6


def make_xk(n_cols):
    rng = np.random.default_rng(0)
    return rng.uniform(0.1, 0.9, size=(NX, n_cols))


def test_observes_only_scim_and_bbs_with_scaled_noise():
    xk = make_xk(3)
    v = np.array([[1.0, 2.0, 3.0],
                   [1.0, 1.0, 1.0]])
    dt = 4.0  # sqrt(dt) = 2, easy to check by hand

    y = measurement(dt, xk, u=None, constants=None, v=v, noise_index=False,
                      augment_states=True, v_from_filter=0)

    expected_noise = np.diag([0.03, 0.05]) @ v * np.sqrt(dt)
    assert y.shape == (2, 3)
    assert np.allclose(y, xk[:2, :] + expected_noise)


def test_v_from_filter_uses_v_directly():
    xk = make_xk(2)
    v = np.array([[0.1, 0.2], [0.3, 0.4]])

    y = measurement(dt=1.0, xk=xk, u=None, constants=None, v=v,
                      noise_index=False, augment_states=True, v_from_filter=1)

    assert np.allclose(y, xk[:2, :] + v)


def test_noise_index_returns_diffusion_column_broadcast():
    xk = make_xk(4)

    y1 = measurement(dt=1.0, xk=xk, u=None, constants=None,
                       v=np.zeros((2, 4)), noise_index=1, augment_states=True,
                       v_from_filter=0)
    y2 = measurement(dt=1.0, xk=xk, u=None, constants=None,
                       v=np.zeros((2, 4)), noise_index=2, augment_states=True,
                       v_from_filter=0)

    assert y1.shape == (2, 4)
    assert np.allclose(y1, np.tile(np.array([[0.03], [0.0]]), (1, 4)))
    assert np.allclose(y2, np.tile(np.array([[0.0], [0.05]]), (1, 4)))


if __name__ == "__main__":
    tests = [test_observes_only_scim_and_bbs_with_scaled_noise,
              test_v_from_filter_uses_v_directly,
              test_noise_index_returns_diffusion_column_broadcast]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
