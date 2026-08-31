"""Checks for who_therapies.py against who_therapies.m."""

import numpy as np

from applications.sci_rehab.who_therapies import who_therapies


def test_shapes_and_counts():
    T = who_therapies()
    assert T.n == 13
    assert len(T.codes) == 13
    assert len(T.names) == 13
    assert len(T.assessment) == 13
    assert T.G.shape == (2, 13)
    assert T.load.shape == (13,)
    assert T.codes == [f"U{i}" for i in range(1, 14)]


def test_known_values_match_matlab_source():
    T = who_therapies()
    # U7 = balance training: high BBS gain (0.35), modest SCIM gain (0.10)
    assert np.isclose(T.G[0, 6], 0.10)  # SCIM
    assert np.isclose(T.G[1, 6], 0.35)  # BBS
    # U3 = muscle-strengthening: high SCIM gain, high physical load
    assert np.isclose(T.G[0, 2], 0.30)
    assert np.isclose(T.load[2], 0.85)
    # U12 = hand/arm training: loads SCIM heavily, almost nothing on balance
    assert np.isclose(T.G[0, 11], 0.26)
    assert np.isclose(T.G[1, 11], 0.02)


def test_fresh_instance_each_call_no_shared_mutable_state():
    T1 = who_therapies()
    T2 = who_therapies()
    T1.G[0, 0] = 999.0
    assert T2.G[0, 0] != 999.0  # mutating T1 must not affect T2


if __name__ == "__main__":
    tests = [test_shapes_and_counts,
              test_known_values_match_matlab_source,
              test_fresh_instance_each_call_no_shared_mutable_state]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
