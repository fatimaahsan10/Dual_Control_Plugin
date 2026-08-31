"""
Explicit constants object for the dynamic-pricing plug-in -- same role as
applications/sci_rehab/sindy_model_constants.py's SindyModelConstants:
carries every plant-specific numeric parameter that dynamics/measurement/
cost need, threaded through the `constants` argument every DynamicsFn/
MeasurementFn/CostFn/ContinuousDynamicsFn implementation accepts (see
extensions/dual_control/protocols.py).

b (price sensitivity, the slope of the linear demand curve D(u;b) = a -
b*u) is the one parameter estimated online via dual iLQG -- see
continuous_dynamics.py's docstring. `b_prior` is used when
augment_states=False (an unaugmented call has no b_hat row in `xa` to
read from) and doubles as the natural initial estimate `p_hat` fed into
ilqg_function()/main_outer_control_loop() when augmented -- same role as
continuous_state_dynamics.py's `constants.Xi` supplying default
coefficient values that augmentation then selectively overrides.
"""

from dataclasses import dataclass


@dataclass
class PricingConstants:
    a: float          # demand intercept at price 0 (known)
    c: float          # marginal cost per unit (known)
    eta: float        # demand partial-adjustment rate, in (0, 1] (known)
    p_ref: float      # reference price for the cost regularizer
    b_prior: float    # nominal/prior price sensitivity (used when augment_states=False)
    F_dyn: float = 1.0       # dynamics noise scale on x
    F_param: float = 0.05    # parameter (b) noise scale -- random-walk diffusion
    sigma_y: float = 1.0     # measurement noise scale on observed demand
    lambda_reg: float = 1e-4  # cost regularizer weight (QP conditioning only, not economic)
