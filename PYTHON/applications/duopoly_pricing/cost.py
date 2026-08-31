"""
Running and terminal cost for duopoly pricing under unknown own- and
cross-price demand sensitivities.

    cost = -(profit_A + profit_B) + resource regularizer ( + 0 at the
            terminal step )
      profit_A  : (p_A - c_A) * D_A(p_A, p_B; b_hat)
      profit_B  : (p_B - c_B) * D_B(p_A, p_B; b_hat)
                  D_A/D_B are the SAME instantaneous demand curves
                  continuous_dynamics.py relaxes each state toward -- NOT
                  the lagged states x1/x2 themselves. This is applied
                  from the start (not discovered by trial and error like
                  dynamic_pricing/cost.py's REVISION note): pricing off a
                  lagged state gives the CURRENT price zero within-period
                  cost, so a receding-horizon controller pins it at the
                  bound regardless of belief -- see dynamic_pricing/cost.py's
                  own documented diagnosis of exactly this failure mode.
      regularizer : lambda_reg_A*(p_A-p_ref_A)^2 + lambda_reg_B*(p_B-p_ref_B)^2
                    -- QP-conditioning term only, not a real economic cost,
                    same role as dynamic_pricing/cost.py's regularizer.
      terminal  : 0 -- no price is chosen at the terminal step

    xa: xa[0:2] = realized demands (unused here, same as
        dynamic_pricing/cost.py's unused lagged x1 -- meaningful only to
        the estimator/dynamics, not the objective); xa[2:6] = b_hat, if
        augmented, read directly to price both current demand curves.
    u : u[0] = p_A, u[1] = p_B

D_A(u;b_hat)/D_B(u;b_hat) are deliberately left UNCLAMPED here (unlike
dynamics.py's x>=0 floor) so they stay smooth/differentiable for iLQG's
finite-difference Jacobians -- a price pushed past either choke price now
shows up as a smooth negative-profit penalty rather than a kink, same
reasoning as dynamic_pricing/cost.py.

Matches CostFn's contract exactly (extensions/dual_control/protocols.py):
    cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants)
    -> (K,) array. u columns may be all-NaN to signal "final cost only";
    cost() zeroes them before use.

`tracking_trajectory` is unused -- this cost tracks a fixed joint-profit
objective, not a time-varying reference trajectory -- kept as an unused
parameter for signature fidelity, same convention as
dynamic_pricing/cost.py's unused `tracking_trajectory`.
"""

import numpy as np


def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
    # Final-step controls arrive as NaN -- zero them so they add no
    # regularizer cost. Copied (not mutated in place), matching
    # dynamic_pricing/cost.py's own convention: the caller's u (e.g.
    # forward_pass.py's u_aug) must not be clobbered.
    u = np.array(u, dtype=float, copy=True)
    final = np.isnan(u[0, :])
    u[:, final] = 0.0

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u_phys = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
                    + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)
    else:
        u_phys = u

    p_A, p_B = u_phys[0, :], u_phys[1, :]
    K = xa.shape[1]

    # b_hat: read from the augmented state rows if present, else fall
    # back to the constants' nominal prior -- same convention as
    # continuous_dynamics.py's augment_states=False branch.
    if xa.shape[0] > 2:
        b_AA, b_AB, b_BA, b_BB = xa[2, :], xa[3, :], xa[4, :], xa[5, :]
    else:
        b_AA = np.full(K, constants.b_prior[0])
        b_AB = np.full(K, constants.b_prior[1])
        b_BA = np.full(K, constants.b_prior[2])
        b_BB = np.full(K, constants.b_prior[3])

    D_A = constants.a_A - b_AA * p_A + b_AB * p_B
    D_B = constants.a_B - b_BB * p_B + b_BA * p_A

    profit_A = (p_A - constants.c_A) * D_A
    profit_B = (p_B - constants.c_B) * D_B

    reg = (constants.lambda_reg_A * (p_A - constants.p_ref_A) ** 2
            + constants.lambda_reg_B * (p_B - constants.p_ref_B) ** 2)

    total = -(profit_A + profit_B) + reg
    total[final] = 0.0
    return total
