"""
Backward pass (Ricatti-Mayne recursion) for iLQG/DDP.

Ported from Todorov's CODES/TODOROV'S CODE/iLQG.m, the local function
`back_pass` (nested inside iLQG.m).
"""

import numpy as np

from core.ddp_solver.box_qp import box_qp


def back_pass(cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu,
              lam, reg_type, lims, u):
    """
    Propagate the value function backward and compute the local
    feedback control law u = u_bar + k + K @ (x - x_bar).

    Shapes (n = state dim, m = control dim, N = number of time points,
    i.e. trajectory length + 1; there are N-1 control steps)
    ----------------------------------------------------------------
    cx   : (n, N)         dCost/dx        (all N points, incl. terminal)
    cu   : (m, N)         dCost/du        (last column unused)
    cxx  : (n, n, N)      d2Cost/dx2
    cxu  : (n, m, N)      d2Cost/dxdu
    cuu  : (m, m, N)      d2Cost/du2
    fx   : (n, n, N)      dDynamics/dx    (last slice unused)
    fu   : (n, m, N)      dDynamics/du    (last slice unused)
    fxx  : (n, n, n, N) or None   d2Dynamics/dx2   (full DDP only)
    fxu  : (n, n, m, N) or None   d2Dynamics/dxdu  (full DDP only)
    fuu  : (n, m, m, N) or None   d2Dynamics/du2   (full DDP only)
    lam      : float    current regularization strength (lambda)
    reg_type : int       1 -> regularize Quu directly
                          2 -> regularize Vxx before it propagates (default choice)
    lims     : control bounds, or None. Either
                 (m, 2)      one fixed box for the whole trajectory
                             (original control-limited DDP; Tassa,
                             Mansard & Todorov 2014).
                 (N-1, m, 2) a per-timestep box, lims[i] applying at
                             control step i (Dastan & Sensinger 2024
                             extension -- see
                             zahid/dynamic_control_bounds.py's
                             build_time_varying_lims(), which produces
                             exactly this shape from a nominal
                             trajectory). The "lims[0,0] > lims[0,1]
                             means unconstrained" sentinel only applies
                             to the (m,2) case; pass lims=None to
                             disable constraints when using per-timestep
                             bounds.
    u        : (m, N-1) current control sequence (only needed if lims is set,
               to express box bounds as deltas from the current control)

    Returns
    -------
    diverge : int   0 on success; otherwise the 1-indexed (MATLAB-style,
                     for easy comparison against the original) time step
                     at which Quu wasn't positive-definite / boxQP failed.
                     Caller should raise lambda and retry.
    Vx  : (n, N)       value function gradient at each time point
    Vxx : (n, n, N)    value function Hessian at each time point
    k   : (m, N-1)     feedforward control term
    K   : (m, n, N-1)  feedback control gain
    dV  : (2,)         [1st-order, 2nd-order] predicted cost change,
                        used later by the line search
    """
    n = cx.shape[0]
    m = cu.shape[0]
    N = cx.shape[1]

    k = np.zeros((m, N - 1))
    K = np.zeros((m, n, N - 1))
    Vx = np.zeros((n, N))
    Vxx = np.zeros((n, n, N))
    dV = np.zeros(2)

    # boundary condition: value-to-go at the last time point = terminal cost
    Vx[:, N - 1] = cx[:, N - 1]
    Vxx[:, :, N - 1] = cxx[:, :, N - 1]

    diverge = 0

    # MATLAB: for i = N-1:-1:1  (1-indexed) <-> Python: i = N-2 down to 0
    for i in range(N - 2, -1, -1):
        Vx_next = Vx[:, i + 1]
        Vxx_next = Vxx[:, :, i + 1]

        # ---- 2nd-order expansion of the action-value function Q(x, u) ----
        Qu = cu[:, i] + fu[:, :, i].T @ Vx_next
        Qx = cx[:, i] + fx[:, :, i].T @ Vx_next
        Qux = cxu[:, :, i].T + fu[:, :, i].T @ Vxx_next @ fx[:, :, i]

        fxuVx = 0.0
        if fxu is not None:
            # contracts the dynamics Hessian d2f/dxdu with Vx over the
            # output (state) axis -- curvature-of-dynamics correction,
            # full-DDP only. Result shape (m, n) to match Qux.
            fxuVx = np.einsum('o,oxu->ux', Vx_next, fxu[:, :, :, i])
            Qux = Qux + fxuVx

        Quu = cuu[:, :, i] + fu[:, :, i].T @ Vxx_next @ fu[:, :, i]
        fuuVx = 0.0
        if fuu is not None:
            fuuVx = np.einsum('o,ouv->uv', Vx_next, fuu[:, :, :, i])
            Quu = Quu + fuuVx

        Qxx = cxx[:, :, i] + fx[:, :, i].T @ Vxx_next @ fx[:, :, i]
        if fxx is not None:
            Qxx = Qxx + np.einsum('o,oxy->xy', Vx_next, fxx[:, :, :, i])

        # ---- regularized versions -- used ONLY to solve for k, K ----
        Vxx_reg = Vxx_next + (lam * np.eye(n) if reg_type == 2 else 0.0)

        Qux_reg = cxu[:, :, i].T + fu[:, :, i].T @ Vxx_reg @ fx[:, :, i]
        if fxu is not None:
            Qux_reg = Qux_reg + fxuVx

        QuuF = cuu[:, :, i] + fu[:, :, i].T @ Vxx_reg @ fu[:, :, i]
        if reg_type == 1:
            QuuF = QuuF + lam * np.eye(m)
        if fuu is not None:
            QuuF = QuuF + fuuVx

        # ---- solve the local quadratic subproblem for the control law ----
        no_limits = lims is None or lims.size == 0 or (
            lims.ndim == 2 and lims[0, 0] > lims[0, 1])

        if no_limits:
            # Cholesky solve of QuuF @ [k_i, K_i] = -[Qu, Qux_reg];
            # failure means QuuF isn't positive-definite -> diverge here.
            try:
                L = np.linalg.cholesky(QuuF)  # QuuF = L @ L.T
            except np.linalg.LinAlgError:
                diverge = i + 1
                return diverge, Vx, Vxx, k, K, dV

            rhs = np.column_stack([Qu, Qux_reg])          # (m, 1+n)
            y = np.linalg.solve(L, rhs)
            kK = -np.linalg.solve(L.T, y)
            k_i = kK[:, 0]
            K_i = kK[:, 1:]
        else:
            lims_i = lims if lims.ndim == 2 else lims[i]
            lower = lims_i[:, 0] - u[:, i]
            upper = lims_i[:, 1] - u[:, i]
            k_i, result, R, free = box_qp(
                QuuF, Qu, lower, upper, k[:, min(i + 1, N - 2)]
            )
            if result < 1:
                diverge = i + 1
                return diverge, Vx, Vxx, k, K, dV

            K_i = np.zeros((m, n))
            if np.any(free):
                K_i[free, :] = -np.linalg.solve(R, np.linalg.solve(R.T, Qux_reg[free, :]))

        # ---- propagate the (unregularized) value function to this step ----
        dV = dV + np.array([k_i @ Qu, 0.5 * k_i @ Quu @ k_i])
        Vx[:, i] = Qx + K_i.T @ Quu @ k_i + K_i.T @ Qu + Qux.T @ k_i
        Vxx[:, :, i] = Qxx + K_i.T @ Quu @ K_i + K_i.T @ Qux + Qux.T @ K_i
        Vxx[:, :, i] = 0.5 * (Vxx[:, :, i] + Vxx[:, :, i].T)  # enforce symmetry

        k[:, i] = k_i
        K[:, :, i] = K_i

    return diverge, Vx, Vxx, k, K, dV
