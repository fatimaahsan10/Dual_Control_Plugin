"""
iLQG/DDP outer loop -- solves the deterministic finite-horizon optimal
control problem

    minimize   sum_i cost(x_i, u_i) + final_cost(x_N)
    subject to x_{i+1} = dynamics(x_i, u_i)

Ported from Todorov's CODES/TODOROV'S CODE/iLQG.m (main function body,
Steps 1-4 of the per-iteration loop). Graphics/plotting from the MATLAB
version are dropped; everything algorithmic is kept.

Tassa, Y., Mansard, N., & Todorov, E. (2014). Control-limited differential
dynamic programming. ICRA 2014.

Optional `constraint_fn` param (Dastan & Sensinger 2024 extension, see
zahid/dynamic_control_bounds.py and zahid/relative_degree_reduction.py):
recomputes a state-dependent box bound each outer iteration instead of
using one fixed `lims` for the whole run. back_pass.py/forward_pass.py
were extended to accept this per-timestep bound shape; box_qp.py is
unchanged, since it already solves an arbitrary box per call.
"""

import numpy as np

from core.ddp_solver.back_pass import back_pass
from core.ddp_solver.forward_pass import forward_pass


def ilqg(step_fn, derivs_fn, x0, u0, lims=None, constraint_fn=None, alphas=None,
          tol_fun=1e-7, tol_grad=1e-4, max_iter=500,
          lambda_init=1.0, dlambda_init=1.0, lambda_factor=1.6,
          lambda_max=1e10, lambda_min=1e-6, reg_type=2, z_min=0.0,
          verbose=1):
    """
    Parameters
    ----------
    step_fn   : callable(x, u, i) -> (x_next, cost), vectorized over a
                (dim, K) batch -- see forward_pass.py for the exact contract.
    derivs_fn : callable(x, u) -> (fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu)
                x is (n, N+1), u is (m, N+1) with the last column all-NaN.
                fxx/fxu/fuu may be None (iLQG rather than full DDP).
                Shapes must match what back_pass.py expects.
    x0        : (n,)      initial state
    u0        : (m, N)    initial control sequence
    lims      : (m, 2) or None      control limits, held fixed for the
                whole run (Tassa et al. 2014's original control-limited
                DDP). Ignored from iteration 1 onward if `constraint_fn`
                is given -- only used for the initial divergence-avoiding
                rollout before the main loop starts.
    constraint_fn : callable(x_traj, u_traj) -> (N, m, 2) ndarray, or None.
                Dastan & Sensinger (2024) extension: recomputes a
                state-dependent box bound once per outer iteration, from
                the CURRENT nominal trajectory (x_traj = x[:, :N],
                u_traj = u), and substitutes it for `lims` in that
                iteration's back_pass/forward_pass calls. Typically built
                by closing over a constraint with
                zahid/dynamic_control_bounds.py's build_time_varying_lims
                (directly, for h(x,u)>=0) or composed with
                zahid/relative_degree_reduction.py's
                state_constraint_to_control_constraint first (for a
                state-only h(x)>=0). Constraints start being enforced
                from the first backward pass, not the initial rollout.
    alphas    : (K,)      backtracking step sizes to try in the line search;
                default 10**linspace(0, -3, 11), matching the MATLAB default

    Returns
    -------
    x, u   : optimal trajectory / control sequence
    L      : (m, n, N)   feedback gains from the last backward pass
    Vx, Vxx : value function gradient/Hessian from the last backward pass
    cost   : (N+1,) cost at each time point along the returned trajectory
    trace  : list of dict, one per iteration, for inspecting convergence
    stop_reason : str describing why the loop terminated
    """
    if alphas is None:
        alphas = 10.0 ** np.linspace(0, -3, 11)

    n = x0.shape[0]
    m, N = u0.shape
    u = u0.copy()

    # --- initial rollout; shrink u until it doesn't blow up ---
    x = cost = None
    for a in alphas:
        xr, ur, cr = forward_pass(x0, a * u, None, None, None, np.array([1.0]),
                                    step_fn, lims)
        if np.all(np.abs(xr) < 1e8):
            x, u, cost = xr[:, :, 0], ur[:, :, 0], cr[:, 0]
            break
    if x is None:
        raise RuntimeError("EXIT: initial control sequence caused divergence")

    lam = lambda_init
    dlambda = dlambda_init
    trace = []
    flg_change = True
    stop_reason = None
    Vx = Vxx = K_gain = dV = None

    for it in range(1, max_iter + 1):
        record = {"iter": it}

        # ====== Step 1: differentiate dynamics & cost along current trajectory
        if flg_change:
            u_aug = np.column_stack([u, np.full((m, 1), np.nan)])
            fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu = derivs_fn(x, u_aug)
            if constraint_fn is not None:
                lims = constraint_fn(x[:, :N], u)
            flg_change = False

        # ====== Step 2: backward pass
        back_pass_done = False
        while not back_pass_done:
            diverge, Vx, Vxx, k, K_gain, dV = back_pass(
                cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu,
                lam, reg_type, lims, u)
            if diverge:
                dlambda = max(dlambda * lambda_factor, lambda_factor)
                lam = max(lam * dlambda, lambda_min)
                if lam > lambda_max:
                    break
                continue
            back_pass_done = True

        # termination on small gradient -- computed unconditionally, as in
        # the MATLAB source; if the backward pass never succeeded, lambda
        # is already >= lambda_max here so this can't spuriously trigger.
        g_norm = np.mean(np.max(np.abs(k) / (np.abs(u) + 1), axis=0))
        record["lambda"] = lam
        record["grad_norm"] = g_norm
        if g_norm < tol_grad and lam < 1e-5:
            dlambda = min(dlambda / lambda_factor, 1 / lambda_factor)
            lam = lam * dlambda * (lam > lambda_min)
            stop_reason = "SUCCESS: gradient norm < tol_grad"
            trace.append(record)
            break

        # ====== Step 3: line search
        fwd_pass_done = False
        if back_pass_done:
            xnew, unew, costnew = forward_pass(
                x0, u, K_gain, x[:, :N], k, alphas, step_fn, lims)
            dcost_per_alpha = cost.sum() - costnew.sum(axis=0)
            w = int(np.argmax(dcost_per_alpha))
            dcost = dcost_per_alpha[w]
            alpha_used = alphas[w]
            expected = -alpha_used * (dV[0] + alpha_used * dV[1])
            z = dcost / expected if expected > 0 else np.sign(dcost)
            if z > z_min:
                fwd_pass_done = True
                x_cand, u_cand, cost_cand = (xnew[:, :, w], unew[:, :, w],
                                               costnew[:, w])

        # ====== Step 4: accept or reject the step
        if fwd_pass_done:
            dlambda = min(dlambda / lambda_factor, 1 / lambda_factor)
            lam = lam * dlambda * (lam > lambda_min)

            x, u, cost = x_cand, u_cand, cost_cand
            flg_change = True

            record.update(dcost=float(dcost), alpha=float(alpha_used),
                           z=float(z), cost=float(cost.sum()))
            trace.append(record)

            if dcost < tol_fun:
                stop_reason = "SUCCESS: cost change < tol_fun"
                break
        else:
            dlambda = max(dlambda * lambda_factor, lambda_factor)
            lam = max(lam * dlambda, lambda_min)
            record["cost"] = float(cost.sum())
            trace.append(record)

            if lam > lambda_max:
                stop_reason = "EXIT: lambda > lambda_max"
                break
    else:
        stop_reason = "EXIT: maximum iterations reached"

    if verbose:
        print(f"{stop_reason}  (iters={trace[-1]['iter']}, "
              f"cost={cost.sum():.6g}, grad_norm={trace[-1]['grad_norm']:.3g})")

    return x, u, K_gain, Vx, Vxx, cost, trace, stop_reason
