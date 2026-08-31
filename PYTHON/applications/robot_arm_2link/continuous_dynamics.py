"""
2-link planar robot arm plant dynamics: the standard rigid-body
manipulator equation for a 2-DOF revolute-revolute arm in a vertical
plane (gravity active), with an unknown link-2 mass estimated via dual
control -- see Spong, Hutchinson & Vidyasagar, "Robot Modeling and
Control", or Slotine & Li, "Applied Nonlinear Control" (the classic
"unknown payload mass" adaptive-robot-control benchmark this plug-in is
modeled on).

STATE VECTOR (nx = 4): x1 = q1 (joint 1 angle, rad), x2 = q2 (joint 2
angle, rad, measured relative to link 1), x3 = w1 = qdot1 (joint 1
angular velocity, rad/s), x4 = w2 = qdot2 (joint 2 angular velocity,
rad/s). No target/reference is part of the state itself -- see
cost.py, which reads constants.q1_target/q2_target directly, same
convention as applications/sci_rehab/l_cost.py's fixed persona ceiling.

ACTION VECTOR (nu = 2): u1 = tau1 (torque applied at joint 1, N*m), u2 =
tau2 (torque applied at joint 2, N*m).

AUGMENTED PARAMETER (augment_states=True): xa[4] = m2 -- link 2's mass,
kg. The ONLY unknown; every other physical constant (l1, l2, lc1, lc2,
m1, I1, I2, g, fv1, fv2) is known and lives in `constants`. m2 was
chosen (over e.g. a friction coefficient) because it multiplies EVERY
term of M(q)/C(q,qdot)/G(q) below, not just one -- a genuinely rich
identification target, and the standard textbook example (an unknown
payload rigidly grasped at the end-effector is dynamically identical to
an unknown effective link-2 mass).

EQUATIONS OF MOTION (manipulator equation):
    M(q) @ qddot + C(q, qdot) @ qdot + G(q) + Fv @ qdot = tau

    M(q) = [ m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2)) + I1 + I2,
             m2*(lc2^2 + l1*lc2*cos(q2)) + I2                          ]
           [ m2*(lc2^2 + l1*lc2*cos(q2)) + I2,
             m2*lc2^2 + I2                                             ]

    C(q, qdot) = [ -m2*l1*lc2*sin(q2)*w2,   -m2*l1*lc2*sin(q2)*(w1+w2) ]
                 [  m2*l1*lc2*sin(q2)*w1,    0                         ]

    G(q) = [ (m1*lc1 + m2*l1)*g*cos(q1) + m2*lc2*g*cos(q1+q2) ]
           [  m2*lc2*g*cos(q1+q2)                              ]

    Fv = diag(fv1, fv2)   (known viscous joint friction, NOT the
                            estimated parameter)

    qddot = M(q)^-1 @ (tau - C(q, qdot) @ qdot - G(q) - Fv @ qdot)

M(q) is inverted analytically per column below (closed-form 2x2 inverse)
rather than via np.linalg.solve in a Python loop over K -- vectorizes
cleanly across the (nxa, K) batch shape every other plant in this
codebase uses, and M(q) is guaranteed invertible for l1, lc2, I2 > 0
(det = I2*(m1*lc1^2 + I1) > 0 even at m2 = 0, so the inverse never blows
up even for an unphysically-small parameter estimate mid-identification
-- see arm_constants.py's m2_floor guardrail in dynamics.py for the
actual physical floor).

NO ANGLE WRAPPING: q1/q2 are left as plain unbounded reals -- fine for a
single bounded reach maneuver (this plug-in's use case), not a
continuously-spinning joint. Deliberately deferred, same "one source of
nonlinearity at a time" scoping as the joint-space-not-Cartesian target
choice (see CLAUDE.md's math-review discussion this plug-in was proposed
against).

GUARDRAIL: none applied here -- the m2 positivity floor lives in
dynamics.py's discrete step (continuous-time derivatives are left
unclamped so they stay smooth for iLQG's finite-difference Jacobians),
same split as applications/duopoly_pricing/continuous_dynamics.py's
demand floor.
"""

import numpy as np


def continuous_dynamics(dt, xa, u, constants, augment_states, u_lims,
                          u_lim_method):
    """
    Returns
    -------
    xdot : (nx [+ n_p], K)   state (and, if augmented, parameter) derivatives
    nx   : 4
    n_p  : 1 if augment_states else 0
    F    : (nx, nx)   dynamics noise scale matrix
    Fp   : (n_p, n_p) parameter noise scale matrix, empty if not augmented
    """
    nx = 4
    K = xa.shape[1]

    if u.shape[1] == 1 and K > 1:
        u = np.tile(u, (1, K))

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
              + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)

    l1, lc1, lc2 = constants.l1, constants.lc1, constants.lc2
    m1, I1, I2, g = constants.m1, constants.I1, constants.I2, constants.g
    fv1, fv2 = constants.fv1, constants.fv2

    q1, q2 = xa[0, :], xa[1, :]
    w1, w2 = xa[2, :], xa[3, :]
    tau1, tau2 = u[0, :], u[1, :]

    if augment_states:
        m2 = xa[4, :]
        n_p = 1
    else:
        m2 = np.full(K, constants.m2_prior)
        n_p = 0

    cos_q2 = np.cos(q2)
    sin_q2 = np.sin(q2)

    M11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos_q2) + I1 + I2
    M12 = m2 * (lc2 ** 2 + l1 * lc2 * cos_q2) + I2
    M22 = m2 * lc2 ** 2 + I2

    C11 = -m2 * l1 * lc2 * sin_q2 * w2
    C12 = -m2 * l1 * lc2 * sin_q2 * (w1 + w2)
    C21 = m2 * l1 * lc2 * sin_q2 * w1

    G1 = (m1 * lc1 + m2 * l1) * g * np.cos(q1) + m2 * lc2 * g * np.cos(q1 + q2)
    G2 = m2 * lc2 * g * np.cos(q1 + q2)

    rhs1 = tau1 - (C11 * w1 + C12 * w2) - G1 - fv1 * w1
    rhs2 = tau2 - (C21 * w1) - G2 - fv2 * w2

    det = M11 * M22 - M12 ** 2
    qddot1 = (M22 * rhs1 - M12 * rhs2) / det
    qddot2 = (-M12 * rhs1 + M11 * rhs2) / det

    f = np.vstack([w1.reshape(1, K), w2.reshape(1, K),
                    qddot1.reshape(1, K), qddot2.reshape(1, K)])

    if augment_states:
        fp = np.zeros((n_p, K))  # parameter dynamics: zero-mean random walk
        xdot = np.vstack([f, fp])
    else:
        xdot = f

    F = np.diag([constants.F_dyn_q, constants.F_dyn_q,
                  constants.F_dyn_w, constants.F_dyn_w])
    Fp = (np.array([[constants.F_param_m2]]) if augment_states
           else np.zeros((0, 0)))

    return xdot, nx, n_p, F, Fp
