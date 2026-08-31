"""
Dual/adaptive-iLQG backward pass: propagates a joint quadratic cost-to-go
in the true state x and the state ESTIMATE x-hat (not the estimation
error) back through time, coupled by an externally-supplied estimator
gain K (from Todorov_estimator.m, not yet ported -- K is taken as given).

Ported from Andrew's CODES/ANDREW'S CODE/backward_pass.m.
Based on Li, W. & Todorov, E. (2007), and Andrew Mathis's PhD thesis
(NOTES/2024_Mathis_Sensinger Carretero_PhD Thesis.pdf), section 2.1.3.3,
eqs. (2.49)-(2.63): the value-function ansatz

    v_k(x, xhat) = 1/2 x'Sx x + 1/2 xhat'Sxh xhat + x'Sxxh xhat
                    + x'sx + xhat'sxh + s0

is propagated backward via three coupled Sx/Sxh/Sxxh recursions instead
of Todorov's single Vxx, because control acts on xhat while cost is paid
on the true x, and the two are correlated through K.

DEVIATION FROM THE LITERAL MATLAB SOURCE: backward_pass.m line 178 reads
`sxh(:, k)` where the thesis (and every other recursion in the same
function) uses index k+1. Since `sxh` is filled in decreasing-k order,
that's a stale zero, silently dropping the `F'K'sxhat_{k+1}` term from
the sx update for the whole trajectory. This port uses the thesis's
sxh_next (k+1), per the correction chosen when this was ported --
flagged here in case MATLAB results are ever compared against this code.
"""

import numpy as np

# boxQP.m is duplicated verbatim across both MATLAB codebases (see
# CLAUDE.md); rather than duplicate the Python port too, reuse
# core/ddp_solver/box_qp.py directly.
from core.ddp_solver.box_qp import box_qp


def backward_pass(A, B, c, Cx, Cu, d, Dx, Du, E, F, K, fxx, fxu, fuu,
                    q0, q, Q, r, R, P, lam, reg_type, lims, u, nv,
                    return_internals=False):
    """
    Shapes (n = augmented state dim, m = control dim, N = time points,
    nw = # dynamics noise channels, nv = # measurement noise channels)
    ----------------------------------------------------------------
    A, B      : (n,n,N), (n,m,N)          dynamics Jacobians
    c, Cx, Cu : (n,nw,N), (n,n,N,nw), (n,m,N,nw)   dynamics noise terms
    d, Dx, Du : (ny,nv,N), (ny,n,N,nv), (ny,m,N,nv) measurement noise terms
    E, F      : (ny,m,N), (ny,n,N)        measurement Jacobians (E unused,
                                            kept for signature fidelity)
    K         : (n,ny,N) estimator gain (from Todorov_estimator.m)
    fxx,fxu,fuu : None (full_DDP path unimplemented, see forward_pass.py)
    q0,q,Q,r,R,P : cost derivatives, as returned by forward_pass.py
    lam, reg_type : regularization strength / scheme (1-4)
    lims      : (m,2) control bounds or None
    u         : (m,N) current nominal controls (box-constraint offset)
    nv        : number of measurement noise channels
    return_internals : if True, also return a dict of Sx/Sxh/Sxxh/sx/sxh/
                        s0/H/H_reg/G_reg/Gx/Gxh/g for testing -- the
                        4-value return matches the MATLAB signature by
                        default.

    Returns
    -------
    diverge : 0 on success; k+1 (1-indexed, MATLAB-style) if Cholesky
              failed; -(k+1) if boxQP failed
    l : (m, N-1)   feedforward control term
    L : (m, n, N-1) feedback gain on the state ESTIMATE
    s0_alpha : (2,) [1st-order, 2nd-order] predicted cost change, for the
               line search
    """
    if fxx is not None or fxu is not None or fuu is not None:
        raise NotImplementedError(
            "full_DDP second-derivative terms not ported (see "
            "forward_pass.py -- this is a dead code path in the MATLAB "
            "source)")

    n = Q.shape[0]
    m = R.shape[0]
    N = q.shape[1]
    nw = c.shape[1]

    l = np.zeros((m, N - 1))
    L = np.zeros((m, n, N - 1))

    sx = np.zeros((n, N))
    sxh = np.zeros((n, N))
    Sx = np.zeros((n, n, N))
    Sxh = np.zeros((n, n, N))
    Sxxh = np.zeros((n, n, N))
    s0 = np.zeros(N)

    s0_alpha = np.zeros(2)

    Sx[:, :, N - 1] = Q[:, :, N - 1]
    sx[:, N - 1] = q[:, N - 1]
    s0[N - 1] = q0[N - 1]
    # Sxh, Sxxh, sxh boundary values are 0 (already initialized)

    diverge = 0
    for k in range(N - 2, -1, -1):
        Sx_n, Sxh_n, Sxxh_n = Sx[:, :, k + 1], Sxh[:, :, k + 1], Sxxh[:, :, k + 1]
        sx_n, sxh_n = sx[:, k + 1], sxh[:, k + 1]

        # ---- dynamics-noise covariance sums (Sigma_i c_i c_i' etc.) ----
        c_Sx_c = 0.0
        Cx_Sx_c = np.zeros(n)
        Cx_Sx_Cx = np.zeros((n, n))
        Cu_Sx_c = np.zeros(m)
        Cu_Sx_Cx = np.zeros((m, n))
        Cu_Sx_Cu = np.zeros((m, m))
        for i in range(nw):
            ci, Cxi, Cui = c[:, i, k], Cx[:, :, k, i], Cu[:, :, k, i]
            c_Sx_c += ci @ Sx_n @ ci
            Cx_Sx_c += Cxi.T @ Sx_n @ ci
            Cx_Sx_Cx += Cxi.T @ Sx_n @ Cxi
            Cu_Sx_c += Cui.T @ Sx_n @ ci
            Cu_Sx_Cx += Cui.T @ Sx_n @ Cxi
            Cu_Sx_Cu += Cui.T @ Sx_n @ Cui

        # ---- measurement-noise covariance sums, filtered through K'Sxh K ----
        Kk, Fk = K[:, :, k], F[:, :, k]
        KSxhK = Kk.T @ Sxh_n @ Kk
        d_K_Sxh_K_d = 0.0
        Dx_K_Sxh_K_d = np.zeros(n)
        Dx_K_Sxh_K_Dx = np.zeros((n, n))
        Du_K_Sxh_K_d = np.zeros(m)
        Du_K_Sxh_K_Dx = np.zeros((m, n))
        Du_K_Sxh_K_Du = np.zeros((m, m))
        for i in range(nv):
            di, Dxi, Dui = d[:, i, k], Dx[:, :, k, i], Du[:, :, k, i]
            d_K_Sxh_K_d += di @ KSxhK @ di
            Dx_K_Sxh_K_d += Dxi.T @ KSxhK @ di
            Dx_K_Sxh_K_Dx += Dxi.T @ KSxhK @ Dxi
            Du_K_Sxh_K_d += Dui.T @ KSxhK @ di
            Du_K_Sxh_K_Dx += Dui.T @ KSxhK @ Dxi
            Du_K_Sxh_K_Du += Dui.T @ KSxhK @ Dui

        Ak, Bk = A[:, :, k], B[:, :, k]
        AKF = Ak - Kk @ Fk

        # ---- unregularized cost/gain terms (eqs. 23-26) ----
        g = r[:, k] + Bk.T @ (sx_n + sxh_n) + Cu_Sx_c + Du_K_Sxh_K_d
        Gx = (P[:, :, k].T + Bk.T @ (Sx_n + Sxxh_n) @ Ak
              + Bk.T @ (Sxh_n + Sxxh_n) @ Kk @ Fk + Cu_Sx_Cx + Du_K_Sxh_K_Dx)
        Gxh = Bk.T @ (Sxh_n + Sxxh_n) @ AKF
        H = (R[:, :, k] + Bk.T @ (Sx_n + Sxh_n + 2 * Sxxh_n) @ Bk
             + Cu_Sx_Cu + Du_K_Sxh_K_Du)
        # H must be symmetric (it's a Hessian of cost-to-go w.r.t. u) but
        # "2*Sxxh" is only a valid stand-in for "Sxxh + Sxxh.T" if Sxxh
        # itself is symmetric, which it isn't (bilinear cross term x'Sxxh
        # xhat, not a quadratic form) -- so H inherits real asymmetry here,
        # not just float noise. Downstream solvers (cholesky, eigh) read
        # only one triangle and would otherwise silently use a different,
        # inconsistent matrix from the one grad/cost actually used.
        H = 0.5 * (H + H.T)

        # ---- regularized versions, used ONLY for the control-law solve ----
        S_reg = Sx_n + Sxh_n + 2 * Sxxh_n + (lam * np.eye(n) if reg_type == 2 else 0.0)
        G_reg = P[:, :, k].T + Bk.T @ S_reg @ Ak + Cu_Sx_Cx + Du_K_Sxh_K_Dx
        H_w_S_reg = R[:, :, k] + Bk.T @ S_reg @ Bk + Cu_Sx_Cu + Du_K_Sxh_K_Du
        H_w_S_reg = 0.5 * (H_w_S_reg + H_w_S_reg.T)  # same reasoning as H above

        if reg_type == 1:
            H_reg = H_w_S_reg + lam * np.eye(m)
        elif reg_type == 3:
            H_reg = H_w_S_reg + (lam - np.min(np.linalg.eigvalsh(H_w_S_reg))) * np.eye(m)
        elif reg_type == 4:
            eigval, eigvec = np.linalg.eigh(H_w_S_reg)
            eigval = np.maximum(eigval, lam)
            H_reg = eigvec @ np.diag(eigval) @ eigvec.T
        else:  # reg_type == 2: already folded into S_reg above
            H_reg = H_w_S_reg

        # ---- solve for the control law ----
        no_limits = lims is None or np.size(lims) == 0 or lims[0, 0] > lims[0, 1]

        if no_limits:
            try:
                Lc = np.linalg.cholesky(H_reg)
            except np.linalg.LinAlgError:
                diverge = k + 1
                return (diverge, l, L, s0_alpha) if not return_internals else (
                    diverge, l, L, s0_alpha, None)

            rhs = np.column_stack([g, G_reg])
            y = np.linalg.solve(Lc, rhs)
            lL = -np.linalg.solve(Lc.T, y)
            l_i, L_i = lL[:, 0], lL[:, 1:]
        else:
            try:
                np.linalg.cholesky(H_reg)
            except np.linalg.LinAlgError:
                diverge = k + 1
                return (diverge, l, L, s0_alpha) if not return_internals else (
                    diverge, l, L, s0_alpha, None)

            lower = lims[:, 0] - u[:, k]
            upper = lims[:, 1] - u[:, k]
            warm_start = l[:, min(k + 1, N - 2)]
            l_i, result, Lc, free = box_qp(H_reg, g, lower, upper, warm_start)
            if result < 1:
                diverge = -(k + 1)
                return (diverge, l, L, s0_alpha) if not return_internals else (
                    diverge, l, L, s0_alpha, None)

            L_i = np.zeros((m, n))
            if np.any(free):
                L_i[free, :] = -np.linalg.solve(Lc, np.linalg.solve(Lc.T, G_reg[free, :]))

        # ---- propagate the (unregularized) cost-to-go ----
        s0_alpha = s0_alpha + np.array([l_i @ g, 0.5 * l_i @ H @ l_i])

        Sx[:, :, k] = (Q[:, :, k] + Ak.T @ Sx_n @ Ak
                        + Fk.T @ Kk.T @ Sxh_n @ Kk @ Fk
                        + 2 * Ak.T @ Sxxh_n @ Kk @ Fk
                        + Cx_Sx_Cx + Dx_K_Sxh_K_Dx)
        Sxh[:, :, k] = AKF.T @ Sxh_n @ AKF + L_i.T @ H @ L_i + L_i.T @ Gxh + Gxh.T @ L_i
        # Sx and Sxh multiply a vector against itself (x'Sx x, xhat'Sxh xhat)
        # so are symmetric in exact arithmetic; enforce it to stop floating-
        # point asymmetry from accumulating over the recursion and later
        # tripping up cholesky (which silently reads only the lower
        # triangle). Sxxh is a genuine cross term (x'Sxxh xhat) and is not
        # expected to be symmetric, so it's left alone -- same convention
        # Todorov's back_pass.py uses for Vxx.
        Sx[:, :, k] = 0.5 * (Sx[:, :, k] + Sx[:, :, k].T)
        Sxh[:, :, k] = 0.5 * (Sxh[:, :, k] + Sxh[:, :, k].T)
        Sxxh[:, :, k] = Fk.T @ Kk.T @ Sxh_n @ AKF + Ak.T @ Sxxh_n @ AKF + Gx.T @ L_i

        # NOTE: sxh_n here (not sxh[:,k]) -- the corrected term, see module
        # docstring re: the MATLAB indexing discrepancy.
        sx[:, k] = (q[:, k] + Ak.T @ sx_n + Fk.T @ Kk.T @ sxh_n
                     + Gx.T @ l_i + Cx_Sx_c + Dx_K_Sxh_K_d)
        sxh[:, k] = AKF.T @ sxh_n + L_i.T @ H @ l_i + L_i.T @ g + Gxh.T @ l_i
        s0[k] = (q0[k] + s0[k + 1] + 0.5 * l_i @ H @ l_i + l_i @ g
                  + 0.5 * (c_Sx_c + d_K_Sxh_K_d))

        l[:, k] = l_i
        L[:, :, k] = L_i

    if not return_internals:
        return diverge, l, L, s0_alpha

    # g/H/H_reg/Gx/Gxh below are whatever the last loop iteration (k=0)
    # computed -- enough to sanity-check the control law at that step.
    internals = dict(Sx=Sx, Sxh=Sxh, Sxxh=Sxxh, sx=sx, sxh=sxh, s0=s0,
                      g=g, H=H, H_reg=H_reg, G_reg=G_reg, Gx=Gx, Gxh=Gxh)
    return diverge, l, L, s0_alpha, internals
