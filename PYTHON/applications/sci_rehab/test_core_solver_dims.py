"""
Confirms backward_pass.py and forward_pass.py -- ported from Andrew's
CODES/ANDREW'S CODE/backward_pass.m and forward_pass.m -- work unchanged
at Katie's problem dimensions (6 states, 13 actions), per the "Reuse
as-is" call in CLAUDE.md's Katie conversion plan: backward_pass.m and
forward_pass.m diff from Andrew's only in comments (confirmed via
Compare-Object on both files -- removed dead commented-out code and a
header attribution line, no logic changes), and neither Python port
hardcodes a state/action dimension anywhere (n/m/nxa/nu are all read off
input array shapes). So this file does NOT re-port anything; it imports
Andrew's modules directly (same reuse pattern backward_pass.py itself
uses for box_qp.py) and re-runs cut-down versions of
andrew/test_backward_pass.py's and andrew/test_forward_pass.py's checks
at n=6/m=13 instead of n=3/m=2, to catch any latent dimension
assumption (e.g. a hardcoded loop bound, a reshape that only works for
square-ish shapes) that a comment-only diff review wouldn't surface.

Katie's actual SCI plant (ContinuousStateDynamics.m / l_cost.m /
Measurement.m) isn't ported yet -- see CLAUDE.md, "NOT YET PORTED" --
so this uses synthetic linear/quadratic stand-ins, same approach
andrew/test_forward_pass.py takes for the still-unported SIDARTHE plant.
"""

import numpy as np

from extensions.dual_control.backward_pass import backward_pass
from extensions.dual_control.forward_pass import forward_pass

# Katie's problem dimensions (SCIM, BBS, AIS, Age, DPI, Caregiver support;
# 13-therapy WHO Package of Interventions action space).
N_STATES = 6
N_ACTIONS = 13


def test_backward_pass_sx_matches_riccati_at_6state_13action():
    rng = np.random.default_rng(0)
    n, m, N, ny = N_STATES, N_ACTIONS, 6, 4

    A0 = 0.5 * rng.standard_normal((n, n))
    B0 = 0.3 * rng.standard_normal((n, m))
    Q0 = A0 @ A0.T + np.eye(n)      # SPD
    R0 = rng.standard_normal((m, m))
    R0 = R0 @ R0.T + np.eye(m)      # SPD

    A = np.repeat(A0[:, :, None], N, axis=2)
    B = np.repeat(B0[:, :, None], N, axis=2)
    Q = np.repeat(Q0[:, :, None], N, axis=2)
    R = np.repeat(R0[:, :, None], N, axis=2)
    P = np.zeros((n, m, N))
    q = np.zeros((n, N))
    r = np.zeros((m, N))
    q0 = np.zeros(N)

    c = np.zeros((n, 0, N))
    Cx = np.zeros((n, n, N, 0))
    Cu = np.zeros((n, m, N, 0))
    d = np.zeros((ny, 0, N))
    Dx = np.zeros((ny, n, N, 0))
    Du = np.zeros((ny, m, N, 0))
    F = np.zeros((ny, n, N))
    E = np.zeros((ny, m, N))
    K = np.zeros((n, ny, N))       # no filtering coupling
    u = np.zeros((m, N - 1))

    diverge, l, L, s0_alpha, internals = backward_pass(
        A, B, c, Cx, Cu, d, Dx, Du, E, F, K, None, None, None,
        q0, q, Q, r, R, P, lam=1.0, reg_type=2, lims=None, u=u, nv=0,
        return_internals=True)

    assert diverge == 0
    assert l.shape == (m, N - 1)
    assert L.shape == (m, n, N - 1)

    Sx_ref = np.zeros((n, n, N))
    Sx_ref[:, :, N - 1] = Q[:, :, N - 1]
    for k in range(N - 2, -1, -1):
        Sx_ref[:, :, k] = Q[:, :, k] + A[:, :, k].T @ Sx_ref[:, :, k + 1] @ A[:, :, k]

    assert np.allclose(internals["Sx"], Sx_ref, atol=1e-8)


def test_backward_pass_respects_13_action_control_limits():
    rng = np.random.default_rng(1)
    n, m, N, ny, nw, nv = N_STATES, N_ACTIONS, 6, 4, 2, 2

    A0 = 0.5 * rng.standard_normal((n, n))
    B0 = 0.3 * rng.standard_normal((n, m))
    Q0 = A0 @ A0.T + np.eye(n)
    R0m = rng.standard_normal((m, m))
    R0 = R0m @ R0m.T + np.eye(m)
    F0 = 0.3 * rng.standard_normal((ny, n))
    K0 = 0.1 * rng.standard_normal((n, ny))

    A = np.repeat(A0[:, :, None], N, axis=2)
    B = np.repeat(B0[:, :, None], N, axis=2)
    Q = np.repeat(Q0[:, :, None], N, axis=2)
    R = np.repeat(R0[:, :, None], N, axis=2)
    P = np.zeros((n, m, N))
    F = np.repeat(F0[:, :, None], N, axis=2)
    E = np.zeros((ny, m, N))
    K = np.repeat(K0[:, :, None], N, axis=2)

    q = 0.1 * rng.standard_normal((n, N))
    r = 0.1 * rng.standard_normal((m, N))
    q0 = np.zeros(N)

    c = 0.05 * rng.standard_normal((n, nw, N))
    Cx = 0.05 * rng.standard_normal((n, n, N, nw))
    Cu = 0.05 * rng.standard_normal((n, m, N, nw))
    d = 0.05 * rng.standard_normal((ny, nv, N))
    Dx = 0.05 * rng.standard_normal((ny, n, N, nv))
    Du = 0.05 * rng.standard_normal((ny, m, N, nv))

    u = np.zeros((m, N - 1))
    # 13 therapy doses, each bounded to [0, 1] (normalised dose fraction)
    lims = np.tile(np.array([[0.0, 1.0]]), (m, 1))

    diverge, l, L, s0_alpha = backward_pass(
        A, B, c, Cx, Cu, d, Dx, Du, E, F, K, None, None, None,
        q0, q, Q, r, R, P, lam=1.0, reg_type=1, lims=lims, u=u, nv=nv)

    assert diverge == 0
    assert l.shape == (m, N - 1)
    assert np.all(l >= lims[:, 0:1] - 1e-8) and np.all(l <= lims[:, 1:2] + 1e-8)


def test_forward_pass_matches_analytic_derivatives_at_6state_13action():
    rng = np.random.default_rng(2)
    nxa, nu, nw, nv, ny = N_STATES, N_ACTIONS, 2, 2, 4
    N = 5
    dt = 1.0  # Katie's session interval (1 week)

    A_true = 0.1 * rng.standard_normal((nxa, nxa))
    B_true = 0.1 * rng.standard_normal((nxa, nu))
    F_true = 0.1 * rng.standard_normal((ny, nxa))
    E_true = 0.1 * rng.standard_normal((ny, nu))

    Cx_list = [0.05 * rng.standard_normal((nxa, nxa)) for _ in range(nw)]
    Cu_list = [0.05 * rng.standard_normal((nxa, nu)) for _ in range(nw)]
    c0_list = [0.05 * rng.standard_normal(nxa) for _ in range(nw)]

    Dx_list = [0.05 * rng.standard_normal((ny, nxa)) for _ in range(nv)]
    Du_list = [0.05 * rng.standard_normal((ny, nu)) for _ in range(nv)]
    d0_list = [0.05 * rng.standard_normal(ny) for _ in range(nv)]

    Q = np.diag(rng.uniform(0.5, 2.0, size=nxa))
    R = np.diag(rng.uniform(0.5, 2.0, size=nu))

    def dynamics(dt, xa, u, constants, w, noise_index, augment_states,
                  w_from_filter, lam, u_lims, u_lim_method):
        if noise_index is False or noise_index == 0:
            return A_true @ xa + B_true @ u
        i = noise_index - 1
        return c0_list[i][:, None] + Cx_list[i] @ xa + Cu_list[i] @ u

    def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                      v_from_filter):
        if noise_index is False or noise_index == 0:
            return F_true @ xa + E_true @ u
        j = noise_index - 1
        return d0_list[j][:, None] + Dx_list[j] @ xa + Du_list[j] @ u

    def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
        u = np.where(np.isnan(u), 0.0, u)
        return 0.5 * np.sum(xa * (Q @ xa), axis=0) + 0.5 * np.sum(u * (R @ u), axis=0)

    xa_bar = rng.standard_normal((nxa, N))
    u_bar = np.column_stack([rng.standard_normal((nu, N - 1)),
                               np.full((nu, 1), np.nan)])

    (A, B, c, Cx, Cu, d, Dx, Du, E, F, fxx, fxu, fuu,
     q0, q, Q_out, r, R_out, P) = forward_pass(
        xa_bar, u_bar, constants=None, dynamics=dynamics,
        measurement=measurement, cost=cost, dt=dt, full_DDP=False,
        nw=nw, nv=nv, ny=ny, augment_states=True, dyn_noise_reg=0.0,
        lam=0.0, tracking_trajectory=None, u_lims=None, u_lim_method=None)

    assert fxx is None and fxu is None and fuu is None
    assert A.shape == (nxa, nxa, N)
    assert B.shape == (nxa, nu, N)

    for k in range(N - 1):
        assert np.allclose(A[:, :, k], A_true, atol=1e-3)
        assert np.allclose(B[:, :, k], B_true, atol=1e-3)
        assert np.allclose(F[:, :, k], F_true, atol=1e-3)
        assert np.allclose(E[:, :, k], E_true, atol=1e-3)

    for k in range(N):
        u_k = np.nan_to_num(u_bar[:, k])
        expected_q0 = dt * (0.5 * xa_bar[:, k] @ Q @ xa_bar[:, k]
                              + 0.5 * u_k @ R @ u_k)
        assert np.isclose(q0[k], expected_q0, atol=1e-3)
        assert np.allclose(Q_out[:, :, k], dt * Q, atol=1e-3)

    for k in range(N - 1):
        assert np.allclose(R_out[:, :, k], dt * R, atol=1e-3)


if __name__ == "__main__":
    tests = [test_backward_pass_sx_matches_riccati_at_6state_13action,
              test_backward_pass_respects_13_action_control_limits,
              test_forward_pass_matches_analytic_derivatives_at_6state_13action]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
