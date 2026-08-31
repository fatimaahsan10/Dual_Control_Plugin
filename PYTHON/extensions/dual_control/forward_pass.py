"""
Differentiate dynamics, measurement, and cost along a nominal trajectory
(dual/adaptive iLQG's "forward pass" step -- despite the name, this is
Todorov's Step 1 "differentiate", not a rollout; see forward_pass.py in
PYTHON/todorov for the actual rollout).

Ported from Andrew's CODES/ANDREW'S CODE/forward_pass.m.
Based on Li, W. & Todorov, E. (2007), and Andrew Mathis's PhD thesis
(NOTES/2024_Mathis_Sensinger Carretero_PhD Thesis.pdf), section 2.1.3.1,
eqs. (2.26)-(2.34).

Callable contracts (mirrors the MATLAB function-handle calling convention,
so plant-specific dynamics/measurement/cost -- SIDARTHE today, SCI later
-- can be swapped in without touching this file):

    dynamics(dt, xa, u, constants, w, noise_index, augment_states,
             w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method)
        -> (nxa, K) array.
        noise_index=False -> deterministic next state f(xa, u).
        noise_index=i (1-indexed) -> i-th column of the dynamics noise
        diffusion matrix, as a function of (xa, u) (this is what gets
        differentiated to obtain the multiplicative-noise Jacobians).

    measurement(dt, xa, u, constants, v, noise_index, augment_states,
                v_from_filter)
        -> (ny, K) array, same noise_index convention as dynamics but for
        the measurement diffusion matrix.

    cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants)
        -> (K,) array. u columns may be all-NaN to signal "final cost
        only"; cost() is responsible for zeroing them before use (matches
        l_cost.m's own `u(:,final) = 0` convention).
"""

import numpy as np

from extensions.dual_control.finite_difference import finite_difference


def forward_pass(xa_bar, u_bar, constants, dynamics, measurement, cost, dt,
                   full_DDP, nw, nv, ny, augment_states, dyn_noise_reg,
                   lam, tracking_trajectory, u_lims, u_lim_method):
    nxa = xa_bar.shape[0]
    nu = u_bar.shape[0]
    N = xa_bar.shape[1]
    ixa = slice(0, nxa)
    iu = slice(nxa, nxa + nu)
    xau_bar = np.vstack([xa_bar, u_bar])

    # ---- dynamics derivatives ----
    def xu_f_dyn(xau):
        return dynamics(dt, xau[ixa, :], xau[iu, :], constants,
                          np.zeros((nw, 1)), False, augment_states, 0,
                          dyn_noise_reg * lam, u_lims, u_lim_method)

    J_f = finite_difference(xu_f_dyn, xau_bar)
    A = J_f[:, ixa, :]
    B = J_f[:, iu, :]

    if full_DDP:
        # Andrew's own iLQG_function.m hardcodes full_DDP=0 and comments
        # this branch as "not incorporated into iLQG changes from iLQR" --
        # it's unexercised, unverified MATLAB code, so it isn't ported.
        raise NotImplementedError(
            "full_DDP second-derivative branch not ported (dead code path "
            "in the MATLAB source -- Main_Outer_Control_Loop.m always sets "
            "full_DDP=0)")
    fxx = fxu = fuu = None

    # ---- dynamics noise Jacobians (one per noise channel i = 1..nw) ----
    c = np.zeros((nxa, nw, N))
    Cx = np.zeros((nxa, nxa, N, nw))
    Cu = np.zeros((nxa, nu, N, nw))
    for i in range(1, nw + 1):
        def xu_F_dyn(xau, i=i):
            return dynamics(dt, xau[ixa, :], xau[iu, :], constants,
                              np.zeros((nw, 1)), i, augment_states, 0,
                              dyn_noise_reg * lam, u_lims, u_lim_method)
        c[:, i - 1, :] = xu_F_dyn(xau_bar)
        J_F = finite_difference(xu_F_dyn, xau_bar)
        Cx[:, :, :, i - 1] = J_F[:, ixa, :]
        Cu[:, :, :, i - 1] = J_F[:, iu, :]

    c = c * np.sqrt(dt)
    Cx = Cx * np.sqrt(dt)
    Cu = Cu * np.sqrt(dt)

    # ---- measurement derivatives ----
    def xu_g_dyn(xau):
        return measurement(dt, xau[ixa, :], xau[iu, :], constants,
                             np.zeros((nv, xau.shape[1])), False,
                             augment_states, 0)

    J_g = finite_difference(xu_g_dyn, xau_bar)
    F = J_g[:, ixa, :]
    E = J_g[:, iu, :]

    # ---- measurement noise Jacobians (one per noise channel j = 1..nv) ----
    d = np.zeros((ny, nv, N))
    Dx = np.zeros((ny, nxa, N, nv))
    Du = np.zeros((ny, nu, N, nv))
    for j in range(1, nv + 1):
        def xu_G_dyn(xau, j=j):
            return measurement(dt, xau[ixa, :], xau[iu, :], constants,
                                 np.zeros((nv, xau.shape[1])), j,
                                 augment_states, 0)
        d[:, j - 1, :] = xu_G_dyn(xau_bar)
        J_G = finite_difference(xu_G_dyn, xau_bar)
        Dx[:, :, :, j - 1] = J_G[:, ixa, :]
        Du[:, :, :, j - 1] = J_G[:, iu, :]

    d = d / np.sqrt(dt)
    Dx = Dx / np.sqrt(dt)
    Du = Du / np.sqrt(dt)

    # ---- cost first derivatives ----
    def xu_cost(xau):
        return cost(xau[ixa, :], xau[iu, :], tracking_trajectory, u_lims,
                     u_lim_method, constants).reshape(1, -1)

    J = finite_difference(xu_cost, xau_bar)[0]  # (nxa+nu, N)
    lx = J[ixa, :]
    lu = J[iu, :]

    # ---- cost second derivatives (Hessian via nested finite difference) ----
    def xu_Jcst(xau):
        return finite_difference(xu_cost, xau)[0]  # (nxa+nu, K)

    JJ = finite_difference(xu_Jcst, xau_bar)          # (nxa+nu, nxa+nu, N)
    JJ = 0.5 * (JJ + JJ.transpose(1, 0, 2))            # symmetrize
    lxx = JJ[ixa, ixa, :]
    lxu = JJ[ixa, iu, :]
    luu = JJ[iu, iu, :]

    # ---- assemble, scale by dt (matches eq. 2.34) ----
    q0 = cost(xa_bar, u_bar, tracking_trajectory, u_lims, u_lim_method,
               constants)
    q, Q, r, R, P = lx, lxx, lu, luu, lxu

    q0 = dt * q0
    q = dt * q
    Q = dt * Q
    r = dt * r
    R = dt * R
    P = dt * P

    return A, B, c, Cx, Cu, d, Dx, Du, E, F, fxx, fxu, fuu, q0, q, Q, r, R, P
