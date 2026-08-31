"""
End-to-end dual-control demo for the dynamic-pricing plug-in: wires the
reused Andrew/Katie solver core (main_outer_control_loop.py, imported
directly -- no duplication) to the pricing plant
(dynamics.py/continuous_dynamics.py/measurement.py/cost.py), starting
from a DELIBERATELY WRONG initial guess for the price sensitivity b.

Same qualitative check as extensions/dual_control/test_main_outer_control_loop.py
("does the estimate move toward the truth") and CLAUDE.md's Andrew STATUS
entry ("deliberately wrong parameter (0.4) correctly converges toward
true value (1.0)") -- this is the pricing plant's version of that same
proof, run through main_outer_control_loop() rather than a synthetic
linear test system, and the first time this plug-in is exercised through
the real online re-plan / step / re-estimate / warm-start MPC loop
rather than just the shape/signature validator.

Confirms two things:
  1. the estimated b_hat converges toward the TRUE b over sessions
     (ends up closer than it started, not just "moved")
  2. the chosen prices stay economically sensible -- inside
     [p_min, p_max], not pinned at either extreme every session

REVISION HISTORY (both are real findings from actually running the
solver end-to-end, not hypothetical -- see CLAUDE.md's dynamic-pricing
demo review for the full diagnosis of each):
  (a) cost.py originally priced profit off the LAGGED demand state x1;
      fixed to price off the CURRENT price-implied demand D(u;b_hat)
      instead, since the lagged-state version gave price zero
      within-period cost, so a receding-horizon controller always
      pinned it at the upper bound regardless of b_hat.
  (b) even after (a), price stayed pinned at the ceiling -- traced to
      u_lim_method=2's tanh-squash bound enforcement: a badly wrong
      initial belief (session 0) pushed the RAW pre-tanh control deep
      into tanh's saturated region, where d(price)/d(u_raw) is
      effectively zero, so gradient-based re-optimization in later
      sessions couldn't pull it back even after b_hat corrected itself
      (warm-starting carries the saturated raw value forward). Switched
      to u_lim_method=1 (box-QP bound enforcement in back_pass.py,
      operating on price directly, no reparameterization) -- this has no
      such saturation region, and price correctly snaps to the true
      monopoly-optimal ~$30 from session 1 onward.

Run directly: python run_pricing_demo.py
"""

import numpy as np

from extensions.dual_control.main_outer_control_loop import main_outer_control_loop

from applications.dynamic_pricing.continuous_dynamics import continuous_dynamics
from applications.dynamic_pricing.cost import cost
from applications.dynamic_pricing.dynamics import dynamics
from applications.dynamic_pricing.measurement import measurement
from applications.dynamic_pricing.pricing_constants import PricingConstants

N_SESSIONS = 15
DT = 1.0
# P_MAX is deliberately kept BELOW the true demand-choke price a/B_TRUE=50
# (see the note in run_demo below) so that even a badly wrong prior
# pushing the optimizer toward the price ceiling doesn't crash true
# demand into its floor clamp -- an initial P_MAX=60 did exactly that,
# and once demand pins at the x>=0 floor the dynamics Jacobian w.r.t. b
# goes to zero there, so the parameter estimate stops learning and just
# random-walks on process noise instead of converging.
P_MIN, P_MAX = 10.0, 45.0

B_TRUE = 2.0    # true price sensitivity (unknown to the seller)
B_PRIOR = 0.5   # deliberately wrong initial guess -- seller believes
                # demand is much LESS price-sensitive than it really is


def run_demo(n_sessions=N_SESSIONS, seed=0, verbose=False):
    rng = np.random.default_rng(seed)  # noqa: F841 (kept for future noise-seeding hooks)

    constants = PricingConstants(
        a=100.0, c=10.0, eta=0.3, p_ref=35.0, b_prior=B_PRIOR,
        F_dyn=1.0, F_param=0.05, sigma_y=1.0, lambda_reg=1e-4)

    # Plausible starting demand: the TRUE equilibrium demand at p_ref
    # under the TRUE b (the seller can observe current sales directly,
    # even without knowing WHY demand responds the way it does).
    x0 = np.array([constants.a - B_TRUE * constants.p_ref])
    p_hat_0 = np.array([B_PRIOR])
    p_true = np.array([B_TRUE])
    cov_X = 1.0               # demand-estimate variance
    cov_P = np.array([[1.0]])  # b-estimate variance -- wide relative to
                                 # |B_TRUE - B_PRIOR| = 1.5, so the prior
                                 # doesn't itself rule out the true value

    u_lims = np.array([[P_MIN, P_MAX]])

    result = main_outer_control_loop(
        T=n_sessions * DT, dt=DT, x_hat_0=x0, x_true_0=x0.copy(),
        p_hat_0=p_hat_0, p_true=p_true, cov_X=cov_X, cov_P=cov_P,
        constants=constants, u_lims=u_lims, u_lim_method=1,
        dynamics=dynamics, measurement=measurement, cost=cost,
        continuous_dynamics=continuous_dynamics, ny=1, nv=1,
        reg_type=1, max_du_iterations=60, first_run_max_du_iterations=100,
        augment_states_in_ilqg=True, augment_states_in_filter=True,
        verbose=verbose)

    return result, constants


def print_report(result):
    x_true, p_hat, u = result["x_true"], result["p_hat"], result["u"]
    N = x_true.shape[1]

    # With u_lim_method=1, bounds are enforced by back_pass.py's box-QP
    # directly on the control -- `u` IS the physical price already, no
    # tanh conversion needed (unlike u_lim_method=2's reparameterization).
    prices = u[0, :]

    print("=== Dynamic pricing dual-control demo ===")
    print(f"True b={B_TRUE}, initial (wrong) estimate b_hat_0={B_PRIOR}\n")
    print(f"{'Session':>7} | {'demand':>8} | {'price':>7} | {'b_hat':>7}")
    print("-" * 40)
    for k in range(N):
        price_str = f"{prices[k]:7.2f}" if k < N - 1 else "   --  "
        print(f"{k:7d} | {x_true[0, k]:8.2f} | {price_str} | {p_hat[0, k]:7.3f}")

    err0 = abs(p_hat[0, 0] - B_TRUE)
    errN = abs(p_hat[0, -1] - B_TRUE)

    print(f"\nb_hat error: session 0 = {err0:.3f}  ->  final session = {errN:.3f}")
    print(f"Price range chosen: [{prices.min():.2f}, {prices.max():.2f}]  "
          f"(bounds were [{P_MIN}, {P_MAX}])")
    print(f"Total true cost over horizon: {result['total_true_cost']:.2f}\n")

    return err0, errN, prices


if __name__ == "__main__":
    result, constants = run_demo(verbose=False)
    print_report(result)
