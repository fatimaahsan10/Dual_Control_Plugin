"""
Outer MPC loop for dual/adaptive control: re-plan (ilqg_function), step the
true system (simulate_system), re-estimate (spkf_function), warm-start,
repeat.

Ported from Andrew's CODES/ANDREW'S CODE/Main_Outer_Control_Loop.m.
This is Andrew's contribution -- it lives in extensions/dual_control/
(not a per-person folder) because its dynamics/measurement/cost/
continuous_dynamics callables make it plant-agnostic, which is exactly
why applications/sci_rehab/run_katie_pipeline.py reuses it unmodified
to drive Katie's SCI plant.

Per outer iteration:
    1. ilqg_function(...) from the current x_hat/p_hat estimate -> a full
       planned trajectory, of which only the first control is applied
       (standard MPC/receding-horizon)
    2. simulate_system(...) advances the TRUE system one real step under
       that control, producing a (possibly noisy) measurement
    3. spkf_function(...) fuses the measurement into an updated state +
       parameter estimate and covariance
    4. the control/gain trajectory is shifted (rolling horizon) or
       truncated (shrinking horizon) to warm-start the next iLQG call

Two covariance trackers are kept deliberately separate, mirroring the
MATLAB source: cov_xa_hat_filter (the outer SPKF's belief) and
cov_xa_hat_iLQG (what's fed into the next inner iLQG call) -- these CAN
differ because augment_states_in_filter and augment_states_in_ilqg are
independent flags.

NOT PORTED: the multi-seed first-iteration search (first_run_seeds > 1 in
the MATLAB source) -- off by default in the reference configuration and
not exercised by it. Requesting it here raises NotImplementedError rather
than silently ignoring the request.

DEVIATION FROM THE LITERAL MATLAB SOURCE: dynamics/measurement/cost/
continuous_dynamics are explicit callables, matching every other module
in this port, rather than hardcoded calls to the SIDARTHE-specific
functions.
"""

import numpy as np

from extensions.dual_control.ilqg_function import ilqg_function
from extensions.dual_control.simulate_system import simulate_system
from extensions.dual_control.spkf_function import spkf_function, _block_diag


def main_outer_control_loop(
        T, dt, x_hat_0, x_true_0, p_hat_0, p_true, cov_X, cov_P, constants,
        u_lims, u_lim_method, dynamics, measurement, cost,
        continuous_dynamics, ny, nv, nw=None,
        lambda_i=1.0, dlambda_i=1.0, reset_lambda=True, dyn_noise_reg=0.0,
        reg_type=3, max_du_iterations=250, first_run_max_du_iterations=500,
        augment_states_in_ilqg=True, augment_states_in_filter=True,
        keep_dyn_noise=False, keep_meas_noise=False,
        keep_dyn_noise_filter=True, keep_meas_noise_filter=True,
        horizon_mode=2, h=np.sqrt(3), additive_noise=False,
        first_run_seeds=1, u_bar_0=None, verbose=False):
    """
    Returns a dict: x_hat, p_hat, x_true, xa_true, u, y_true, cost_true,
    cost_est (all (dim, N) or (N,) shaped -- see body), total_true_cost.

    u_bar_0 : optional (nu, N) initial control guess for the FIRST iLQG
        call only (every later call warm-starts from the previous
        solution's shifted trajectory regardless). Defaults to None
        (all-zeros, original behaviour) -- added to let a caller nudge a
        non-convex cost landscape's first solve away from the all-zero
        local optimum that a bilinear (cross-therapy) cost surface can
        otherwise settle into; not in the original MATLAB source.
    """
    if first_run_seeds > 1:
        raise NotImplementedError(
            "multi-seed first iteration not ported -- off (first_run_seeds"
            "=1) in the reference MATLAB configuration and not exercised "
            "by it")

    N = round(T / dt)
    nx = x_hat_0.shape[0]
    n_p = p_hat_0.shape[0]
    nu = u_lims.shape[0]

    if nw is None:
        nw = nx + n_p if augment_states_in_ilqg else nx

    ix = np.arange(nx)

    if augment_states_in_filter:
        ip = np.arange(n_p) + nx
        constants_filter = constants
        cov_xa_hat_filter = _block_diag(cov_X * np.eye(nx), cov_P)
        n_xa_true = nx + n_p
    else:
        ip = None
        constants_filter = np.concatenate([constants, p_hat_0])
        cov_xa_hat_filter = cov_X * np.eye(nx)
        n_xa_true = nx

    cov_xa_hat_ilqg = (_block_diag(cov_X * np.eye(nx), cov_P)
                        if augment_states_in_ilqg else cov_X * np.eye(nx))

    sqrtQ_true = np.linalg.cholesky(np.eye(n_xa_true))
    sqrtR_true = np.linalg.cholesky(np.eye(nv))

    T_rem = (np.arange(T, -dt, -dt) if horizon_mode == 1
              else T * np.ones(N + 1))

    lam, dlambda = lambda_i, dlambda_i

    u_bar = np.zeros((nu, N)) if u_bar_0 is None else np.array(u_bar_0, dtype=float)
    l = np.zeros((nu, N))
    nxa_ilqg = nx + n_p if augment_states_in_ilqg else nx
    L = np.zeros((nu, nxa_ilqg, N))

    x_hat = np.zeros((nx, N))
    x_hat[:, 0] = x_hat_0
    p_hat = np.zeros((n_p, N))
    p_hat[:, 0] = p_hat_0
    x_true = np.zeros((nx, N))
    x_true[:, 0] = x_true_0
    xa_true = np.zeros((n_xa_true, N))
    xa_true[:, 0] = (np.concatenate([x_true_0, p_true])
                       if augment_states_in_filter else x_true_0)

    u = np.zeros((nu, N - 1))
    y_true = np.zeros((ny, N - 1))
    cost_true = np.zeros(N)
    cost_est = np.zeros(N)

    for iter_outer in range(N - 1):
        if verbose:
            print(f"Progress: Step #{iter_outer + 1} of {N}")

        # ---- 1. re-plan ----
        max_iters = (first_run_max_du_iterations if iter_outer == 0
                      else max_du_iterations)
        xnew, unew, l, L, _, _, _, Pw, Pv, converged = ilqg_function(
            T_rem[iter_outer], dt, x_hat[:, iter_outer], l, L, u_bar, lam,
            dlambda, constants, p_hat[:, iter_outer], cov_xa_hat_ilqg,
            augment_states_in_ilqg, reg_type, u_lims, ny, nv, nw, max_iters,
            dyn_noise_reg, None, u_lim_method, dynamics, measurement, cost,
            simulate_system, verbose=False)
        u[:, iter_outer] = unew[:, 0]

        if reset_lambda:
            lam, dlambda = lambda_i, dlambda_i

        # ---- 2. step the true system one real step ----
        y_sim, xa_sim = simulate_system(
            dt, xa_true[:, iter_outer], u[:, iter_outer:iter_outer + 1],
            constants, keep_meas_noise * sqrtR_true,
            keep_dyn_noise * sqrtQ_true, augment_states_in_filter, 0.0,
            u_lims, u_lim_method, dynamics, measurement)
        xa_true[:, iter_outer + 1] = xa_sim[:, -1]
        y_true[:, iter_outer] = y_sim[:, -1]
        x_true[:, iter_outer + 1] = xa_true[ix, iter_outer + 1]

        cost_true[iter_outer] = cost(
            xa_true[:, iter_outer + 1:iter_outer + 2],
            u[:, iter_outer:iter_outer + 1], None, u_lims, u_lim_method,
            constants)[0]

        # ---- 3. re-estimate via SPKF ----
        if augment_states_in_filter:
            xa_hat = np.concatenate([x_hat[:, iter_outer], p_hat[:, iter_outer]])
            _, _, _, _, Fp = continuous_dynamics(
                dt, xa_hat.reshape(-1, 1), u[:, iter_outer:iter_outer + 1],
                constants_filter, augment_states_in_filter, u_lims,
                u_lim_method)
            Pw[nx:nx + n_p, nx:nx + n_p, :] = Fp[0, 0]
        else:
            xa_hat = x_hat[:, iter_outer]

        u_applied = u[:, iter_outer]

        def dynamics_f(dt_, xk, wk, u_applied=u_applied):
            return dynamics(dt_, xk, u_applied[:, None], constants_filter,
                              wk, 0, augment_states_in_filter, 1, 0.0,
                              u_lims, u_lim_method)

        def measurement_f(xk, vk, u_applied=u_applied):
            return measurement(dt, xk, u_applied[:, None], constants_filter,
                                 vk, 0, augment_states_in_filter, 1)

        xa_hat_filtered_all, P_x_filtered = spkf_function(
            dt, dynamics_f, measurement_f,
            y_true[:, iter_outer].reshape(1, -1), xa_hat,
            np.zeros_like(xa_hat), np.zeros(ny), cov_xa_hat_filter,
            Pw * keep_dyn_noise_filter, Pv * keep_meas_noise_filter, h,
            additive_noise)
        xa_hat_filtered = xa_hat_filtered_all[:, 0]

        x_hat[:, iter_outer + 1] = xa_hat_filtered[ix]
        cost_est[iter_outer] = cost(
            xa_hat_filtered.reshape(-1, 1), u[:, iter_outer:iter_outer + 1],
            None, u_lims, u_lim_method, constants)[0]

        # ---- 4. warm start for the next iLQG call ----
        if horizon_mode == 1:
            l = l[:, 1:]
            L = L[:, :, 1:]
            u_bar = unew[:, 1:]
        else:
            u_bar = np.column_stack([unew[:, 1:], unew[:, -1:]])

        if augment_states_in_filter:
            p_hat[:, iter_outer + 1] = xa_hat_filtered[ip]
        else:
            p_hat[:, iter_outer + 1] = p_hat[:, iter_outer]

        cov_xa_hat_filter = P_x_filtered
        cov_xa_hat_ilqg = (cov_xa_hat_filter if augment_states_in_ilqg
                            else cov_xa_hat_filter[:nx, :nx])

    # ---- final time-step costs (terminal, NaN control) ----
    last = N - 1
    nan_u = np.full((nu, 1), np.nan)
    xa_hat_last = (np.concatenate([x_hat[:, last], p_hat[:, last]])
                    if augment_states_in_filter else x_hat[:, last])
    cost_est[last] = cost(xa_hat_last.reshape(-1, 1), nan_u, None, u_lims,
                            u_lim_method, constants)[0]
    cost_true[last] = cost(xa_true[:, last:last + 1], nan_u, None, u_lims,
                             u_lim_method, constants)[0]

    return dict(x_hat=x_hat, p_hat=p_hat, x_true=x_true, xa_true=xa_true,
                 u=u, y_true=y_true, cost_true=cost_true, cost_est=cost_est,
                 total_true_cost=cost_true.sum())
