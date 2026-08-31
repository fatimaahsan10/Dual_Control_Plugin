"""
Running and terminal cost for the 2-link robot arm reaching task.

    cost = state-tracking cost + resource cost ( + terminal boost at
            the final step )
      state    : weighted squared gap of q1/q2 from the fixed
                 joint-space target (constants.q1_target/q2_target),
                 plus a weighted squared penalty on w1/w2 -- keeps the
                 arm from swinging wildly en route, not just penalizing
                 the endpoint
      resource : cu1*tau1^2 + cu2*tau2^2 -- "applying torque costs
                 something", same role as
                 applications/sci_rehab/l_cost.py's w_resource term
      terminal : same state-gap terms, larger weights (see
                 arm_constants.py's cx*_terminal/cv*_terminal), applied
                 only at the final step -- this is what makes "reach the
                 target" mean SETTLE there at rest, not merely pass
                 through it

    xa: xa[0]=q1  xa[1]=q2  xa[2]=w1  xa[3]=w2  [xa[4]=m2, if augmented,
        unused here -- the target is fixed, not m2-dependent]
    u : u[0]=tau1  u[1]=tau2

Matches CostFn's contract exactly (extensions/dual_control/protocols.py):
    cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants)
    -> (K,) array. u columns may be all-NaN to signal "final cost only";
    cost() zeroes them before use.

`tracking_trajectory` is unused -- this cost tracks a FIXED joint-space
target, not a time-varying reference trajectory -- kept as an unused
parameter for signature fidelity, same convention as
applications/duopoly_pricing/cost.py's unused `tracking_trajectory`.
"""

import numpy as np


def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
    # Final-step controls arrive as NaN -- zero them so they add no
    # resource cost. Copied (not mutated in place): the caller's u
    # (e.g. forward_pass.py's u_aug) must not be clobbered.
    u = np.array(u, dtype=float, copy=True)
    final = np.isnan(u[0, :])
    u[:, final] = 0.0

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u_phys = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
                    + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)
    else:
        u_phys = u

    q1, q2, w1, w2 = xa[0, :], xa[1, :], xa[2, :], xa[3, :]
    tau1, tau2 = u_phys[0, :], u_phys[1, :]

    gap1 = q1 - constants.q1_target
    gap2 = q2 - constants.q2_target

    lx = constants.cx1 * gap1 ** 2 + constants.cx2 * gap2 ** 2
    lv = constants.cv1 * w1 ** 2 + constants.cv2 * w2 ** 2
    lu = constants.cu1 * tau1 ** 2 + constants.cu2 * tau2 ** 2

    lf = np.zeros(xa.shape[1])
    lf[final] = (constants.cx1_terminal * gap1[final] ** 2
                  + constants.cx2_terminal * gap2[final] ** 2
                  + constants.cv1_terminal * w1[final] ** 2
                  + constants.cv2_terminal * w2[final] ** 2)

    # Terminal step adds lf ON TOP of the running state/velocity cost
    # (not a replacement) -- same convention as
    # applications/sci_rehab/l_cost.py's lx+lu+lf. lu is already 0 at
    # the terminal step (u zeroed above).
    return lx + lv + lu + lf
