"""
Checks for finite_difference.py and forward_pass.py against hand-derived
analytic values, using synthetic linear dynamics/measurement and quadratic
cost that match the MATLAB calling contract (see forward_pass.py docstring)
without depending on the SIDARTHE-specific DiscreteStateDynamics.m /
Measurement.m / l_cost.m, which are plant-specific and not ported here.
"""

import numpy as np

from extensions.dual_control.finite_difference import finite_difference
from extensions.dual_control.forward_pass import forward_pass


def test_finite_difference_recovers_linear_map():
    rng = np.random.default_rng(0)
    n, K = 4, 6
    A = rng.standard_normal((n, n))

    def fun(x):
        return A @ x

    x = rng.standard_normal((n, K))
    J = finite_difference(fun, x)  # (n, n, K)

    for k in range(K):
        assert np.allclose(J[:, :, k], A, atol=1e-4)


def make_synthetic_system(rng, nxa, nu, nw, nv, ny):
    A_true = 0.1 * rng.standard_normal((nxa, nxa))
    B_true = 0.1 * rng.standard_normal((nxa, nu))
    F_true = 0.1 * rng.standard_normal((ny, nxa))
    E_true = 0.1 * rng.standard_normal((ny, nu))

    # per-noise-channel affine maps c_i(xa,u) = c0_i + Cx_i @ xa + Cu_i @ u
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

    truth = dict(A=A_true, B=B_true, F=F_true, E=E_true,
                  Cx_list=Cx_list, Cu_list=Cu_list, c0_list=c0_list,
                  Dx_list=Dx_list, Du_list=Du_list, d0_list=d0_list,
                  Q=Q, R=R)
    return dynamics, measurement, cost, truth


def test_forward_pass_matches_analytic_derivatives():
    rng = np.random.default_rng(1)
    nxa, nu, nw, nv, ny = 3, 2, 2, 2, 2
    N = 5  # trajectory length (N-1 control steps + 1 terminal)
    dt = 0.1

    dynamics, measurement, cost, truth = make_synthetic_system(
        rng, nxa, nu, nw, nv, ny)

    xa_bar = rng.standard_normal((nxa, N))
    u_bar = np.column_stack([rng.standard_normal((nu, N - 1)),
                               np.full((nu, 1), np.nan)])

    (A, B, c, Cx, Cu, d, Dx, Du, E, F, fxx, fxu, fuu,
     q0, q, Q, r, R, P) = forward_pass(
        xa_bar, u_bar, constants=None, dynamics=dynamics,
        measurement=measurement, cost=cost, dt=dt, full_DDP=False,
        nw=nw, nv=nv, ny=ny, augment_states=True, dyn_noise_reg=0.0,
        lam=0.0, tracking_trajectory=None, u_lims=None, u_lim_method=None)

    assert fxx is None and fxu is None and fuu is None

    # dynamics/measurement Jacobians: check non-terminal columns only
    # (final column is NaN-poisoned by design -- back_pass never reads it,
    # matching Todorov's back_pass.py convention).
    for k in range(N - 1):
        assert np.allclose(A[:, :, k], truth["A"], atol=1e-3)
        assert np.allclose(B[:, :, k], truth["B"], atol=1e-3)
        assert np.allclose(F[:, :, k], truth["F"], atol=1e-3)
        assert np.allclose(E[:, :, k], truth["E"], atol=1e-3)
        for i in range(nw):
            # Cx/Cu/c all pick up the sqrt(dt) scaling forward_pass.py applies
            # to match eqs. (2.26)-(2.27): ci=sqrt(dt)F^[i], Cx_i=sqrt(dt)*dF^[i]/dx
            assert np.allclose(Cx[:, :, k, i], np.sqrt(dt) * truth["Cx_list"][i], atol=1e-3)
            assert np.allclose(Cu[:, :, k, i], np.sqrt(dt) * truth["Cu_list"][i], atol=1e-3)
            expected_c = (truth["c0_list"][i] + truth["Cx_list"][i] @ xa_bar[:, k]
                          + truth["Cu_list"][i] @ u_bar[:, k]) * np.sqrt(dt)
            assert np.allclose(c[:, i, k], expected_c, atol=1e-3)
        for j in range(nv):
            # Dx/Du pick up the 1/sqrt(dt) scaling from eq. (2.27)
            assert np.allclose(Dx[:, :, k, j], truth["Dx_list"][j] / np.sqrt(dt), atol=1e-3)
            assert np.allclose(Du[:, :, k, j], truth["Du_list"][j] / np.sqrt(dt), atol=1e-3)
            expected_d = (truth["d0_list"][j] + truth["Dx_list"][j] @ xa_bar[:, k]
                          + truth["Du_list"][j] @ u_bar[:, k]) / np.sqrt(dt)
            assert np.allclose(d[:, j, k], expected_d, atol=1e-3)

    # cost derivatives: state-cost terms (q0, q, Q) are well-defined at every
    # column, including the terminal one, since cost() zeroes NaN controls
    # itself. Control-cost curvature (R) is NOT meaningful at the terminal
    # column: perturbing a NaN control still gets zeroed by cost(), so the
    # finite difference sees no change and R comes out 0 there by
    # construction -- exactly like back_pass.py's "last slice unused"
    # convention for fx/fu, just for a different reason (NaN-zeroing rather
    # than an unused array slot).
    for k in range(N):
        u_k = np.nan_to_num(u_bar[:, k])
        expected_q0 = dt * (0.5 * xa_bar[:, k] @ truth["Q"] @ xa_bar[:, k]
                              + 0.5 * u_k @ truth["R"] @ u_k)
        assert np.isclose(q0[k], expected_q0, atol=1e-3)
        assert np.allclose(q[:, k], dt * truth["Q"] @ xa_bar[:, k], atol=1e-3)
        assert np.allclose(r[:, k], dt * truth["R"] @ u_k, atol=1e-3)
        assert np.allclose(Q[:, :, k], dt * truth["Q"], atol=1e-3)
        assert np.allclose(P[:, :, k], 0.0, atol=1e-3)  # no cross term in test cost

    for k in range(N - 1):
        assert np.allclose(R[:, :, k], dt * truth["R"], atol=1e-3)


if __name__ == "__main__":
    tests = [test_finite_difference_recovers_linear_map,
              test_forward_pass_matches_analytic_derivatives]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
