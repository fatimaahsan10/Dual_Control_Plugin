"""
Running and terminal cost for dynamic pricing under unknown demand
sensitivity.

    cost = -profit + resource regularizer ( + 0 at the terminal step )
      profit    : (price - marginal_cost) * D(price; b_hat), where
                  D(u;b) = a - b*u is the SAME instantaneous demand curve
                  continuous_dynamics.py relaxes the state toward -- NOT
                  the lagged state x1 itself.
      regularizer : lambda_reg * (price - p_ref)^2 -- QP-conditioning
                    term only (same role as
                    applications/sci_rehab/l_cost.py's 1e-6*sum(u^2)
                    term), not a real economic cost
      terminal  : 0 -- no price is chosen at the terminal step

    xa: xa[0] = realized demand x1 (unused here -- see REVISION note
        below); xa[1] = b_hat, if augmented, read directly to price the
        current demand curve
    u : xa[0] = price

REVISION (was: profit = (price - c) * x1, the LAGGED demand state):
under the original formulation, current-period profit depended on
demand ALREADY realized before this period's price was chosen, so
raising price had zero cost to the CURRENT period's payoff -- only a
future-period consequence via the dynamics. Under the receding-horizon
MPC in main_outer_control_loop.py, which only ever executes the first
action of each freshly re-planned trajectory, that meant price was
pinned at the upper bound every single session (confirmed end-to-end via
run_pricing_demo.py, and confirmed NOT fixable by increasing lambda_reg
-- even lambda_reg=10, 100000x the originally agreed value, only pulled
price to ~$35 instead of the true monopoly-optimal ~$30). Pricing off
D(u;b_hat) instead gives the CURRENT action a real, differentiable,
within-period cost to raising price -- the fix is structural, not a
regularizer tuning problem. See CLAUDE.md's dynamic-pricing demo review
for the full diagnosis.

One consequence: xa's demand row x1 is no longer read by cost() at all
-- it remains meaningful as the estimator's observation target
(measurement.py) and as what dynamics.py evolves, but the objective is
now a function of price and the CURRENT belief about b, not of realized
sales. D(u;b_hat) is deliberately left UNCLAMPED here (unlike
continuous_dynamics.py's x>=0 floor) so it stays smooth/differentiable
for iLQG's finite-difference Jacobians; a price pushed past the choke
price a/b now shows up as a smooth negative-profit penalty rather than a
kink, which is exactly what discourages the optimizer from choosing it.

Matches CostFn's contract exactly (extensions/dual_control/protocols.py):
    cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants)
    -> (K,) array. u columns may be all-NaN to signal "final cost only";
    cost() zeroes them before use.

`tracking_trajectory` is unused -- this cost tracks a fixed profit
objective, not a time-varying reference trajectory -- kept as an unused
parameter for signature fidelity, same convention as
applications/sci_rehab/l_cost.py's unused `tracking_trajectory`.
"""

import numpy as np


def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
    # Final-step controls arrive as NaN -- zero them so they add no
    # regularizer cost. Copied (not mutated in place), matching
    # applications/sci_rehab/l_cost.py's own convention: the caller's u
    # (e.g. forward_pass.py's u_aug) must not be clobbered.
    u = np.array(u, dtype=float, copy=True)
    final = np.isnan(u[0, :])
    u[:, final] = 0.0

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u_phys = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
                    + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)
    else:
        u_phys = u

    price = u_phys[0, :]
    K = xa.shape[1]

    # b_hat: read from the augmented state row if present, else fall
    # back to the constants' nominal prior -- same convention as
    # continuous_dynamics.py's augment_states=False branch.
    if xa.shape[0] > 1:
        b_hat = xa[1, :]
    else:
        b_hat = np.full(K, constants.b_prior)

    demand_at_price = constants.a - b_hat * price
    profit = (price - constants.c) * demand_at_price
    reg = constants.lambda_reg * (price - constants.p_ref) ** 2

    total = -profit + reg
    total[final] = 0.0
    return total
