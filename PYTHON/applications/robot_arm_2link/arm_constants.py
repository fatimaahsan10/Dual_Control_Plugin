"""
Explicit constants object for the 2-link planar robot arm plug-in --
same role as applications/duopoly_pricing/duopoly_constants.py's
DuopolyPricingConstants.

PHYSICAL PARAMETERS (known): l1/l2 (link lengths), lc1/lc2 (distance from
each joint to its own link's center of mass), m1 (link 1 mass), I1/I2
(link moments of inertia about their own center of mass), g (gravity),
fv1/fv2 (known joint viscous friction). See continuous_dynamics.py's
module docstring for how these combine into M(q)/C(q,qdot)/G(q).

m2_prior is link 2's mass -- the ONE parameter estimated online via dual
iLQG (see continuous_dynamics.py). Same dual role as
DuopolyPricingConstants.b_prior: used directly when augment_states=False
(an unaugmented call has no m2_hat row in `xa` to read from), and doubles
as the natural initial estimate p_hat_0 fed into main_outer_control_loop()
when augmented.

TARGET: q1_target/q2_target -- the joint-space configuration the arm is
being driven to (see cost.py). Cartesian (end-effector x/y) targets are a
deliberately deferred follow-up, not implemented here -- see the
CLAUDE.md math-review discussion this plug-in was proposed against.

COST WEIGHTS: cx1/cx2 (running position-tracking weight), cv1/cv2
(running velocity weight -- keeps the trajectory from swinging wildly en
route, not just penalizing the endpoint), cu1/cu2 (running torque-effort
weight), and the terminal counterparts (cx*_terminal/cv*_terminal),
applied only at the final step with larger weights so the arm actually
settles AT the target at rest, not just passes through it -- same
running-vs-terminal-weight-boost convention as
applications/sci_rehab/l_cost.py's w_scim/w_terminal_scim split.

NOISE SCALES: F_dyn_q/F_dyn_w (dynamics noise scale on the position vs.
velocity rows of the physical state -- kept separate, not a single
shared F_dyn, since q1/q2 are purely kinematically integrated from
w1/w2 while w1/w2 carry the real unmodeled-torque/model-mismatch
uncertainty), F_param_m2 (m2's own zero-drift random-walk noise scale,
same convention as DuopolyPricingConstants.F_param_A/_B),
sigma_y_q/sigma_y_w (measurement noise scale on the position vs.
velocity rows of the observation -- see measurement.py).

m2_floor: small positive floor applied to m2 after each Euler step in
dynamics.py, purely to keep the augmented state physically valid (mass
can't go negative) if the estimate's random walk ever wanders close to
zero -- same "guardrail" convention as
applications/duopoly_pricing/dynamics.py's x>=0 demand floor.
"""

from dataclasses import dataclass


@dataclass
class RobotArmConstants:
    l1: float             # link 1 length, m (known)
    l2: float             # link 2 length, m (known)
    lc1: float            # joint-1-to-link-1-COM distance, m (known)
    lc2: float            # joint-2-to-link-2-COM distance, m (known)
    m1: float              # link 1 mass, kg (known)
    I1: float              # link 1 moment of inertia about its own COM, kg*m^2 (known)
    I2: float              # link 2 moment of inertia about its own COM, kg*m^2 (known)
    g: float                # gravitational acceleration, m/s^2 (known)
    fv1: float             # joint 1 viscous friction coefficient, N*m*s/rad (known)
    fv2: float             # joint 2 viscous friction coefficient, N*m*s/rad (known)
    m2_prior: float        # nominal/prior link 2 mass, kg (used when augment_states=False)

    q1_target: float       # target joint 1 angle, rad
    q2_target: float       # target joint 2 angle, rad

    cx1: float = 10.0             # running cost weight on (q1 - q1_target)^2
    cx2: float = 10.0             # running cost weight on (q2 - q2_target)^2
    cv1: float = 1.0              # running cost weight on w1^2
    cv2: float = 1.0              # running cost weight on w2^2
    cu1: float = 0.01             # running cost weight on tau1^2
    cu2: float = 0.01             # running cost weight on tau2^2

    cx1_terminal: float = 200.0   # terminal cost weight on (q1 - q1_target)^2
    cx2_terminal: float = 200.0   # terminal cost weight on (q2 - q2_target)^2
    cv1_terminal: float = 50.0    # terminal cost weight on w1^2 (drives the arm to rest)
    cv2_terminal: float = 50.0    # terminal cost weight on w2^2 (drives the arm to rest)

    F_dyn_q: float = 1e-4         # dynamics noise scale on q1, q2
    F_dyn_w: float = 0.05         # dynamics noise scale on w1, w2
    F_param_m2: float = 0.01      # noise scale for m2's random-walk drift

    sigma_y_q: float = 0.02       # measurement noise scale on observed q1, q2
    sigma_y_w: float = 0.05       # measurement noise scale on observed w1, w2

    m2_floor: float = 0.05        # positive floor on m2 after each Euler step
