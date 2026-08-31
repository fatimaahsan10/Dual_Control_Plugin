"""
Dual/adaptive iLQG inner loop: wires forward_pass (differentiate),
Todorov_estimator (estimator gain K), and backward_pass together with a
line search, the same differentiate/backward-pass/line-search/accept-reject
structure as Todorov's ilqg.py, extended for estimation.

Ported from Andrew's CODES/ANDREW'S CODE/iLQG_function.m.

Simplifications made relative to the literal MATLAB source (all dead code,
verified not to affect any returned value -- see module discussion):
  - the line search's x_hat/y/eta bookkeeping is dropped: computed via K/F/E
    inside the MATLAB loop but never read afterward (not returned, not used
    in the accept/reject decision)
  - the redundant post-loop simulate_system call that recomputes xnew's
    last column (already computed identically inside the loop, given the
    same deterministic zero-noise inputs) is dropped
  - plotting-only bookkeeping (costs_saved, lambdas_saved, U_old) is dropped,
    matching how Todorov's ilqg.py already drops MATLAB's graphics() calls

DEVIATION FROM THE LITERAL MATLAB SOURCE: dynamics/measurement/cost/
simulate_system are explicit callables here rather than hardcoded calls to
the SIDARTHE-specific functions, matching forward_pass.py's/
todorov_estimator.py's/simulate_system.py's own convention.
"""

import numpy as np

from extensions.dual_control.backward_pass import backward_pass
from extensions.dual_control.forward_pass import forward_pass
from extensions.dual_control.todorov_estimator import todorov_estimator


def ilqg_function(T, dt, x0, l, L, u_bar, lam, dlambda, constants, p_hat,
                    cov_xa_hat_0, augment_states, reg_type, u_lims, ny, nv,
                    nw, max_du_iterations, dyn_noise_reg, tracking_trajectory,
                    u_lim_method, dynamics, measurement, cost,
                    simulate_system_fn, verbose=False):
    """
    Returns
    -------
    xa0      : final state trajectory, (nxa, N+1)
    u_bar    : locally optimal control trajectory, (nu, N)
    l, L     : final control policy gains
    lam, dlambda : final regularization values
    cost_arr : cost along the final trajectory, (N+1,)
    Pw, Pv   : realized dynamics/measurement noise covariance snapshots
               along the accepted line-search trajectory (nxa,nxa,N) /
               (ny,ny,N) -- consumed by the OUTER loop's filter, not used
               here
    converged : 0 not converged, 1 gradient, 2 cost, 3 max iters,
                -1 lambda > lambda_max
    """
    full_DDP = False
    dlambda_0 = 1.6
    lambda_max = 1e12
    lambda_min = 1e-6
    alphas = 10.0 ** np.linspace(0, -3, 11)
    tol_fun = 1e-4
    tol_grad = 1e-3
    z_min = 0.0

    nu, N = u_bar.shape
    nx = x0.shape[0]
    n_p = p_hat.shape[0]
    nxa = nx + n_p if augment_states else nx

    if augment_states:
        xa0 = np.concatenate([x0, p_hat])
    else:
        xa0 = x0.copy()
        constants = np.concatenate([constants, p_hat])

    sqrtR = np.linalg.cholesky(np.eye(nv)) if nv > 0 else np.zeros((0, 0))
    sqrtQ = np.linalg.cholesky(np.eye(nw)) if nw > 0 else np.zeros((0, 0))

    j = 1
    flg_change = True
    cost_arr = None
    u_lims_boxqp = u_lims if u_lim_method == 1 else None
    converged = 0
    Pw = Pv = None

    while j < max_du_iterations:
        # ---- roll out nominal trajectory ----
        if xa0.ndim == 1:
            y_bar, xa_bar = simulate_system_fn(
                dt, xa0, u_bar, constants, 0.0 * sqrtR, 0.0 * sqrtQ,
                augment_states, dyn_noise_reg * lam, u_lims, u_lim_method,
                dynamics, measurement)
        elif xa0.shape[1] == N + 1:
            xa_bar = xa0
            u_aug = np.column_stack([u_bar, np.full((nu, 1), np.nan)])
            y_bar = measurement(dt, xa0, u_aug, constants,
                                  np.zeros((nv, N + 1)), False,
                                  augment_states, 0)
        else:
            raise ValueError("xa0 incorrect size")

        # ---- forward pass: differentiate dynamics/measurement/cost ----
        if flg_change:
            u_aug = np.column_stack([u_bar, np.full((nu, 1), np.nan)])
            (A, B, c, Cx, Cu, d, Dx, Du, E, F, fxx, fxu, fuu,
             q0, q, Q, r, R, P) = forward_pass(
                xa_bar, u_aug, constants, dynamics, measurement, cost, dt,
                full_DDP, nw, nv, ny, augment_states, dyn_noise_reg, lam,
                tracking_trajectory, u_lims, u_lim_method)
            cost_arr = cost(xa_bar, u_aug, tracking_trajectory, u_lims,
                              u_lim_method, constants)
            flg_change = False
            if verbose:
                print("Forward pass complete")

        # ---- estimator: produces K for the backward pass ----
        (_, _, K, m_x, m_e, Sigma_x, Sigma_e, Sigma_xe,
         Sigma_ex) = todorov_estimator(
            A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, xa_bar, y_bar, u_bar,
            np.zeros(nxa), cov_xa_hat_0, dt, constants, sqrtR,
            augment_states, measurement)
        if verbose:
            print("Estimator complete")

        # ---- backward pass ----
        back_pass_done = False
        while not back_pass_done:
            diverge, l, L, s0_a = backward_pass(
                A, B, c, Cx, Cu, d, Dx, Du, E, F, K, fxx, fxu, fuu,
                q0, q, Q, r, R, P, lam, reg_type, u_lims_boxqp, u_bar, nv)
            if diverge != 0:
                dlambda = max(dlambda * dlambda_0, dlambda_0)
                lam = max(lam * dlambda, lambda_min)
                if verbose:
                    if diverge > 0:
                        print(f"Cholesky failed at timestep {diverge}.")
                    else:
                        print(f"BoxQP failed at timestep {-diverge}.")
                if lam > lambda_max:
                    break
                continue
            back_pass_done = True
        if verbose and back_pass_done:
            print("Backward pass complete")

        # termination on small gradient -- computed unconditionally (matches
        # MATLAB / Todorov's ilqg.py: harmless when back_pass_done is False
        # since lambda is already >= lambda_max there)
        g_norm = np.mean(np.max(np.abs(l) / (np.abs(u_bar) + 1), axis=0))
        if g_norm < tol_grad and lam < 1e-5:
            dlambda = min(dlambda / dlambda_0, 1 / dlambda_0)
            lam = lam * dlambda * (lam > lambda_min)
            converged = 1
            if verbose:
                print("SUCCESS: gradient norm < tol_grad")
            break

        # ---- line search ----
        fwd_pass_done = False
        alpha = np.nan
        for alpha in alphas:
            du = np.zeros((nu, N))
            unew = np.zeros((nu, N))
            x = np.zeros((nxa, N + 1))
            xnew = np.zeros((nxa, N + 1))
            Pw = np.zeros((nxa, nxa, N))
            Pv = np.zeros((ny, ny, N))

            xnew[:, 0] = xa0 if xa0.ndim == 1 else xa0[:, 0]
            for k in range(N):
                du[:, k] = alpha * l[:, k] + L[:, :, k] @ (xnew[:, k] - xa_bar[:, k])
                unew[:, k] = u_bar[:, k] + du[:, k]

                _, xa_sim = simulate_system_fn(
                    dt, xnew[:, k], unew[:, k:k + 1], constants,
                    0.0 * sqrtR, 0.0 * sqrtQ, augment_states,
                    dyn_noise_reg * lam, u_lims, u_lim_method,
                    dynamics, measurement)
                xnew[:, k + 1] = xa_sim[:, -1]

                Cmat = np.zeros((nxa, nw))
                for i in range(nw):
                    Cmat[:, i] = (c[:, i, k] + Cx[:, :, k, i] @ x[:, k]
                                   + Cu[:, :, k, i] @ du[:, k])
                    Pw[:, :, k] += np.outer(Cmat[:, i], Cmat[:, i])

                Dmat = np.zeros((ny, nv))
                for i in range(nv):
                    Dmat[:, i] = (d[:, i, k] + Dx[:, :, k, i] @ x[:, k]
                                   + Du[:, :, k, i] @ du[:, k])
                    Pv[:, :, k] += np.outer(Dmat[:, i], Dmat[:, i])

                zeta = sqrtQ @ np.random.standard_normal(nw)
                x[:, k + 1] = A[:, :, k] @ x[:, k] + B[:, :, k] @ du[:, k] + Cmat @ zeta

            unew_aug = np.column_stack([unew, np.full((nu, 1), np.nan)])
            costnew = cost(xnew, unew_aug, tracking_trajectory, u_lims,
                             u_lim_method, constants)

            dcost = cost_arr.sum() - costnew.sum()
            expected = -alpha * (s0_a[0] + alpha * s0_a[1])
            z = dcost / expected if expected > 0 else np.sign(dcost)
            if verbose:
                print(f"alpha={alpha:.4g} dcost={dcost:.4g} expected={expected:.4g} z={z:.4g}")
            if z > z_min:
                fwd_pass_done = True
                break

        # ---- accept or reject ----
        if fwd_pass_done:
            dlambda = min(dlambda / dlambda_0, 1 / dlambda_0)
            lam = lam * dlambda * (lam > lambda_min)

            u_bar = unew[:, :N]
            xa0 = xnew
            cost_arr = costnew
            flg_change = True

            if dcost < tol_fun:
                converged = 2
                if verbose:
                    print("SUCCESS: cost change < tol_fun")
                break
        else:
            dlambda = max(dlambda * dlambda_0, dlambda_0)
            lam = max(lam * dlambda, lambda_min)

            if lam > lambda_max:
                lam, dlambda = 1.0, 1.0
                converged = -1
                if verbose:
                    print("FINISHED: lambda > lambda_max")
                break

        j += 1

    if j == max_du_iterations and converged == 0:
        converged = 3
        if verbose:
            print("FINISHED: Max iterations reached")

    return xa0, u_bar, l, L, lam, dlambda, cost_arr, Pw, Pv, converged
