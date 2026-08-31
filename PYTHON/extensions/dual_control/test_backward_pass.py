"""
Checks for backward_pass.py.

Todorov_estimator.m (source of the K gain) isn't ported yet, so a full
closed-loop dual-control verification isn't possible today. What IS
independently checkable without it:

  1. The Sx recursion (eq. 2.50) has no dependence on regularization,
     control cost, or the feedback gain at all -- when there's no noise
     and K=0, it collapses exactly to Sx_k = Q_k + A_k'Sx_{k+1}A_k, a
     plain backward Riccati-style recursion with a known closed form
     that can be computed independently and compared bit-for-bit.
  2. Whatever control law comes out must be a stationary point of its
     own regularized local quadratic model: H_reg @ l = -g (unconstrained
     case). This catches sign/transpose/solve-order bugs regardless of
     whether the "science" (K, noise) is realistic yet.
  3. With box constraints, the returned l must stay feasible.
"""

import numpy as np

from extensions.dual_control.backward_pass import backward_pass


def test_sx_matches_plain_riccati_when_no_noise_no_filter():
    rng = np.random.default_rng(0)
    n, m, N, ny = 3, 2, 6, 2

    A0 = 0.5 * rng.standard_normal((n, n))
    B0 = 0.3 * rng.standard_normal((n, m))
    Q0 = A0 @ A0.T + np.eye(n)      # SPD
    R0 = 0.5 * (rng.standard_normal((m, m)))
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
    K = np.zeros((n, ny, N))       # no filtering coupling at all
    u = np.zeros((m, N - 1))

    diverge, l, L, s0_alpha, internals = backward_pass(
        A, B, c, Cx, Cu, d, Dx, Du, E, F, K, None, None, None,
        q0, q, Q, r, R, P, lam=1.0, reg_type=2, lims=None, u=u, nv=0,
        return_internals=True)

    assert diverge == 0

    Sx_ref = np.zeros((n, n, N))
    Sx_ref[:, :, N - 1] = Q[:, :, N - 1]
    for k in range(N - 2, -1, -1):
        Sx_ref[:, :, k] = Q[:, :, k] + A[:, :, k].T @ Sx_ref[:, :, k + 1] @ A[:, :, k]

    assert np.allclose(internals["Sx"], Sx_ref, atol=1e-8)


def make_random_dual_system(rng, n, m, N, ny, nw, nv):
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

    return A, B, c, Cx, Cu, d, Dx, Du, E, F, K, q0, q, Q, r, R, P


def test_control_law_is_stationary_point_of_its_own_model():
    rng = np.random.default_rng(1)
    n, m, N, ny, nw, nv = 3, 2, 6, 2, 2, 2

    (A, B, c, Cx, Cu, d, Dx, Du, E, F, K,
     q0, q, Q, r, R, P) = make_random_dual_system(rng, n, m, N, ny, nw, nv)
    u = np.zeros((m, N - 1))

    diverge, l, L, s0_alpha, internals = backward_pass(
        A, B, c, Cx, Cu, d, Dx, Du, E, F, K, None, None, None,
        q0, q, Q, r, R, P, lam=1.0, reg_type=1, lims=None, u=u, nv=nv,
        return_internals=True)

    assert diverge == 0
    # unconstrained control law is l = -H_reg^-1 g, L = -H_reg^-1 G_reg
    # (both evaluated at k=0, the last loop iteration -- see internals).
    # H_reg is explicitly symmetrized inside backward_pass.py (it would
    # otherwise inherit real, not just floating-point, asymmetry from
    # Sxxh -- a bilinear cross term with no reason to be symmetric), so
    # this should hold to near machine precision.
    assert np.allclose(internals["H_reg"] @ l[:, 0], -internals["g"], atol=1e-8)
    assert np.allclose(internals["H_reg"] @ L[:, :, 0], -internals["G_reg"], atol=1e-8)


def test_respects_control_limits():
    rng = np.random.default_rng(2)
    n, m, N, ny, nw, nv = 3, 2, 6, 2, 2, 2

    (A, B, c, Cx, Cu, d, Dx, Du, E, F, K,
     q0, q, Q, r, R, P) = make_random_dual_system(rng, n, m, N, ny, nw, nv)
    u = np.zeros((m, N - 1))
    lims = np.array([[-0.3, 0.3], [-0.3, 0.3]])

    diverge, l, L, s0_alpha = backward_pass(
        A, B, c, Cx, Cu, d, Dx, Du, E, F, K, None, None, None,
        q0, q, Q, r, R, P, lam=1.0, reg_type=1, lims=lims, u=u, nv=nv)

    assert diverge == 0
    assert np.all(l >= lims[:, 0:1] - 1e-8) and np.all(l <= lims[:, 1:2] + 1e-8)


if __name__ == "__main__":
    tests = [test_sx_matches_plain_riccati_when_no_noise_no_filter,
              test_control_law_is_stationary_point_of_its_own_model,
              test_respects_control_limits]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
