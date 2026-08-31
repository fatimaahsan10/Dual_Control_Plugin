"""
Explicit constants object for the duopoly-pricing plug-in -- same role as
applications/dynamic_pricing/pricing_constants.py's PricingConstants,
generalized from one seller to two.

b_prior (the 4-vector [b_AA, b_AB, b_BA, b_BB]) is the SET of parameters
estimated online via dual iLQG -- see continuous_dynamics.py's docstring
for the fixed ordering convention every plant file agrees on. `b_prior`
is used when augment_states=False (an unaugmented call has no b_hat rows
in `xa` to read from) and doubles as the natural initial estimate
`p_hat_0` fed into main_outer_control_loop() when augmented -- same dual
role as PricingConstants.b_prior.

Every economic knob that PricingConstants shared between the (only) one
seller is now PER-SELLER here (a, c, eta, p_ref, F_dyn, sigma_y,
lambda_reg all split into _A/_B), per the 2026-08-26 design discussion --
sellers are allowed to differ in cost structure, adjustment speed,
measurement noise, and regularization weight. F_param is split by which
seller's demand EQUATION a parameter feeds into (F_param_A scales
b_AA/b_AB's drift, F_param_B scales b_BA/b_BB's drift), not by which
seller's OWN price the parameter multiplies -- see continuous_dynamics.py.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DuopolyPricingConstants:
    a_A: float          # seller A demand intercept at price 0 (known)
    a_B: float          # seller B demand intercept at price 0 (known)
    c_A: float          # seller A marginal cost per unit (known)
    c_B: float          # seller B marginal cost per unit (known)
    eta_A: float        # seller A demand partial-adjustment rate, in (0, 1] (known)
    eta_B: float        # seller B demand partial-adjustment rate, in (0, 1] (known)
    p_ref_A: float       # reference price for seller A's cost regularizer
    p_ref_B: float       # reference price for seller B's cost regularizer
    b_prior: np.ndarray  # (4,) nominal/prior [b_AA, b_AB, b_BA, b_BB]
                          # (used when augment_states=False)
    F_dyn_A: float = 1.0       # dynamics noise scale on x_A
    F_dyn_B: float = 1.0       # dynamics noise scale on x_B
    F_param_A: float = 0.05    # noise scale for b_AA, b_AB (seller A's equation)
    F_param_B: float = 0.05    # noise scale for b_BA, b_BB (seller B's equation)
    sigma_y_A: float = 1.0     # measurement noise scale on observed demand D_A
    sigma_y_B: float = 1.0     # measurement noise scale on observed demand D_B
    lambda_reg_A: float = 1e-4  # seller A cost regularizer weight (QP conditioning only)
    lambda_reg_B: float = 1e-4  # seller B cost regularizer weight (QP conditioning only)
