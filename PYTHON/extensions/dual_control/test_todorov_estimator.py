"""
Checks for todorov_estimator.py.

No independent "ground truth" estimator exists to cross-check against (this
is a fairly specialized extended-LQG filter), so verification leans on
what's exactly derivable from the recursion's own structure:

  1. m_e (mean of the estimation error) starts at 0 (eq. 2.40) and its
     recursion is homogeneous (m_e_{k+1} = (A-KF) m_e_k, no forcing term)
     -- so it must stay identically 0 for the whole trajectory, regardless
     of A/B/K/L/noise. A clean, always-true invariant.
  2. K_k is *defined* by K_k (F Sigma_e F' + P_k) = A Sigma_e F' (eq. 2.36)
     -- this must hold to near machine precision independent of whether
     the rest of the "science" is realistic, and catches solve-order bugs.
     Uses return_internals=True to get P_k directly rather than working
     around not having it.
  3. m_x's recursion is fully deterministic given K -- independently
     re-simulating it in the test and comparing catches indexing/transpose
     bugs in that part of the loop.
"""

import numpy as np

from extensions.dual_control.todorov_estimator import todorov_estimator


def make_synthetic_system(rng, nx, nu, ny, nw, nv, N):
    A0 = 0.5 * rng.standard_normal((nx, nx))
    B0 = 0.3 * rng.standard_normal((nx, nu))
    F0 = 0.4 * rng.standard_normal((ny, nx))
    E0 = 0.2 * rng.standard_normal((ny, nu))

    A = np.repeat(A0[:, :, None], N, axis=2)
    B = np.repeat(B0[:, :, None], N, axis=2)
    F = np.repeat(F0[:, :, None], N, axis=2)
    E = np.repeat(E0[:, :, None], N, axis=2)

    c = 0.05 * rng.standard_normal((nx, nw, N))
    Cx = 0.05 * rng.standard_normal((nx, nx, N, nw))
    Cu = 0.05 * rng.standard_normal((nx, nu, N, nw))
    d = 0.05 * rng.standard_normal((ny, nv, N))
    Dx = 0.05 * rng.standard_normal((ny, nx, N, nv))
    Du = 0.05 * rng.standard_normal((ny, nu, N, nv))

    l = 0.1 * rng.standard_normal((nu, N - 1))
    L = 0.1 * rng.standard_normal((nu, nx, N - 1))

    x_bar = 0.1 * rng.standard_normal((nx, N))
    y_bar = 0.1 * rng.standard_normal((ny, N))
    u_bar = 0.1 * rng.standard_normal((nu, N - 1))

    def measurement(dt, xa, u, constants, v, noise_index, augment_states, v_from_filter):
        return F0 @ xa + E0 @ u + v

    return A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar, u_bar, measurement


def run_estimator(seed=0, nx=3, nu=2, ny=2, nw=2, nv=2, N=6):
    rng = np.random.default_rng(seed)
    (A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar, u_bar,
     measurement) = make_synthetic_system(rng, nx, nu, ny, nw, nv, N)

    x_hat_0 = 0.1 * rng.standard_normal(nx)
    cov_x_hat_0 = np.eye(nx) * 0.5
    sqrtR = np.eye(nv)

    (x_hat, pi_ctrl, K, m_x, m_e, Sigma_x, Sigma_e, Sigma_xe, Sigma_ex,
     internals) = todorov_estimator(
        A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar, u_bar,
        x_hat_0, cov_x_hat_0, dt=0.1, constants=None, sqrtR=sqrtR,
        augment_states=True, measurement=measurement, return_internals=True)

    return dict(A=A, B=B, F=F, E=E, l=l, L=L, x_hat_0=x_hat_0, N=N,
                 x_hat=x_hat, pi_ctrl=pi_ctrl, K=K, m_x=m_x, m_e=m_e,
                 Sigma_x=Sigma_x, Sigma_e=Sigma_e, Sigma_xe=Sigma_xe,
                 Sigma_ex=Sigma_ex, P=internals["P"], M=internals["M"])


def test_m_e_stays_identically_zero():
    ctx = run_estimator(seed=0)
    assert np.allclose(ctx["m_e"], 0.0, atol=1e-10)


def test_gain_satisfies_its_defining_equation():
    ctx = run_estimator(seed=1)
    K, F, A, Sigma_e, P, N = (ctx["K"], ctx["F"], ctx["A"], ctx["Sigma_e"],
                                ctx["P"], ctx["N"])
    for k in range(N - 1):
        Fk, Ak, Se, Pk = F[:, :, k], A[:, :, k], Sigma_e[:, :, k], P[:, :, k]
        lhs = K[:, :, k] @ (Fk @ Se @ Fk.T + Pk)
        rhs = Ak @ Se @ Fk.T
        assert np.allclose(lhs, rhs, atol=1e-8), f"gain equation violated at k={k}"


def test_m_x_recursion_matches_independent_resimulation():
    ctx = run_estimator(seed=2)
    A, B, F, L, l, K, N = (ctx["A"], ctx["B"], ctx["F"], ctx["L"], ctx["l"],
                             ctx["K"], ctx["N"])

    m_x_ref = np.zeros_like(ctx["m_x"])
    m_x_ref[:, 0] = ctx["x_hat_0"]
    for k in range(N - 1):
        ABL = A[:, :, k] + B[:, :, k] @ L[:, :, k]
        # m_e is identically 0 (checked separately), so the KF@m_e term drops
        m_x_ref[:, k + 1] = ABL @ m_x_ref[:, k] + B[:, :, k] @ l[:, k]

    assert np.allclose(ctx["m_x"], m_x_ref, atol=1e-8)


if __name__ == "__main__":
    tests = [test_m_e_stays_identically_zero,
              test_gain_satisfies_its_defining_equation,
              test_m_x_recursion_matches_independent_resimulation]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
