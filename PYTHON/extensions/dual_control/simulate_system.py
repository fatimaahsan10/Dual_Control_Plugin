"""
Nonlinear stochastic rollout: applies a control sequence through the real
(non-linearized) dynamics/measurement, sampling genuine noise at each step.

Ported from Andrew's CODES/ANDREW'S CODE/simulate_system.m.

DEVIATION FROM THE LITERAL MATLAB SOURCE: MATLAB hardcodes calls to the
SIDARTHE-specific `DiscreteStateDynamics`/`Measurement`. This port takes
them as explicit callables instead (same contract as forward_pass.py),
so it isn't tied to the unported plant code. Also: MATLAB builds y as
(M+1, ny) and transposes it back to (ny, M+1) right before returning to
the caller; this port returns (ny, M+1) directly.
"""

import numpy as np


def simulate_system(dt, xa0, U, constants, sqrtR, sqrtQ, augment_states,
                      lam, u_lims, u_lim_method, dynamics, measurement):
    """
    xa0 : (nx,)     initial state
    U   : (nu, M)   control sequence (M >= 1)

    Returns
    -------
    y  : (ny, M+1)
    xa : (nx, M+1)
    """
    nv = sqrtR.shape[0]
    nw = sqrtQ.shape[0]
    nx = xa0.shape[0]

    U_ext = np.column_stack([U, U[:, -1:]])  # repeat last control once
    steps = U_ext.shape[1] - 1  # = original M

    xa = np.zeros((nx, steps + 1))
    xa[:, 0] = xa0

    v0 = sqrtR @ np.random.standard_normal(nv)
    y0 = measurement(dt, xa0.reshape(-1, 1), U_ext[:, 0:1], constants,
                       v0.reshape(-1, 1), False, augment_states, 0)
    ny = y0.shape[0]
    y = np.zeros((ny, steps + 1))
    y[:, 0] = y0.reshape(-1)

    for i in range(steps):
        w = sqrtQ @ np.random.standard_normal(nw)
        x_next = dynamics(dt, xa[:, i:i + 1], U_ext[:, i:i + 1], constants,
                            w.reshape(-1, 1), False, augment_states, 0, lam,
                            u_lims, u_lim_method)
        xa[:, i + 1] = x_next.reshape(-1)

        v = sqrtR @ np.random.standard_normal(nv)
        y_next = measurement(dt, xa[:, i + 1:i + 2], U_ext[:, i + 1:i + 2],
                               constants, v.reshape(-1, 1), False,
                               augment_states, 0)
        y[:, i + 1] = y_next.reshape(-1)

    return y, xa
