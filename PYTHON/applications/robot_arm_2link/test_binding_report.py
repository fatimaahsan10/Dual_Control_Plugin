"""
Tests for run_robot_arm_demo.py's constraint (tau1 speed derating) and its
binding_report()/print_binding_report() diagnostic -- see CLAUDE.md's
"validating constraint satisfaction" discussion: a trajectory that stays
far below a bound the whole run hasn't actually exercised the constraint
mechanism, only one that repeatedly gets pushed up against its own bound
has. These tests confirm the diagnostic itself reports that distinction
correctly, on both a real end-to-end solve (constraint genuinely binds)
and hand-built synthetic results (constraint never binds / never even
reaches TAU1_DERATE_CAP -- to exercise the "no evidence" branch, which the
real demo's own trajectory never takes).
"""

import numpy as np
import pytest

from applications.robot_arm_2link.run_robot_arm_demo import (
    TAU1_DERATE_CAP, TAU1_DERATE_K, TAU1_LIM, TAU2_LIM,
    binding_report, print_binding_report, run_demo,
)

_U_LIMS = np.array([[-TAU1_LIM, TAU1_LIM], [-TAU2_LIM, TAU2_LIM]])


def test_constrained_run_genuinely_binds():
    result, _ = run_demo(verbose=False, use_constraint=True)
    report = binding_report(result, _U_LIMS)

    assert report["n_tightened"] > 0
    assert report["min_slack_tightened"] is not None
    # session 0 starts at w1=0, so tau1 should be pinned almost exactly at
    # the derating cap (28.0) -- same closed-form check as
    # wizard/test_robot_arm_example.py's own binding test.
    assert report["tau1"][0] == pytest.approx(TAU1_DERATE_CAP, abs=0.05)
    assert report["tau1"].max() < TAU1_LIM


def test_unconstrained_run_would_have_exceeded_the_formula():
    """The counterfactual half of the check: without the constraint, the
    controller's own natural torque choice should cross the same derating
    formula at least once -- direct evidence the constraint is a real
    limiter, not a vacuous one."""
    result_unconstrained, _ = run_demo(verbose=False, use_constraint=False)
    report = binding_report(result_unconstrained, _U_LIMS)
    assert report["n_exceeds_formula"] > 0
    assert report["max_excess"] > 0.0


def test_binding_report_reports_no_evidence_on_a_synthetic_slack_trajectory():
    """Hand-built result where tau1 never gets anywhere near the derating
    bound (even though the bound is still tighter than the flat +-30 N*m
    limit everywhere, since TAU1_DERATE_CAP=28 < 30) -- min_slack should
    be large, and fraction_touching should be 0."""
    N = 10
    x_true = np.zeros((4, N + 1))  # w1 stays 0 -> allowance == TAU1_DERATE_CAP everywhere
    u = np.zeros((2, N))
    u[0, :] = 1.0  # tau1 chosen far below the 28.0 N*m allowance
    result = {"x_true": x_true, "u": u}

    report = binding_report(result, _U_LIMS)
    assert report["n_tightened"] == N  # cap (28) is always < flat limit (30)
    assert report["min_slack_tightened"] == pytest.approx(TAU1_DERATE_CAP - 1.0)
    assert report["fraction_touching"] == 0.0
    assert report["n_exceeds_formula"] == 0


def test_print_binding_report_does_not_raise(capsys):
    result, _ = run_demo(n_sessions=6, verbose=False, use_constraint=True)
    result_unconstrained, _ = run_demo(n_sessions=6, verbose=False, use_constraint=False)
    print_binding_report(
        binding_report(result, _U_LIMS),
        binding_report(result_unconstrained, _U_LIMS))
    out = capsys.readouterr().out
    assert "Constraint binding check" in out
    assert "Unconstrained counterfactual" in out
