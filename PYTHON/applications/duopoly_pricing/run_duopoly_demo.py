"""
End-to-end dual-control demo for the duopoly-pricing plug-in: wires the
reused Andrew/Katie solver core (main_outer_control_loop.py, imported
directly -- no duplication) to the duopoly plant
(dynamics.py/continuous_dynamics.py/measurement.py/cost.py), starting
from DELIBERATELY WRONG initial guesses for all four price sensitivities.

Direct 2-seller generalization of applications/dynamic_pricing/run_pricing_demo.py,
applying both lessons that demo learned the hard way, from the start:
  (a) cost.py prices off the CURRENT price-implied demand D(u;b_hat), not
      the lagged state -- see cost.py's docstring.
  (b) u_lim_method=1 (box-QP bound enforcement in back_pass.py, operating
      on price directly) is used instead of u_lim_method=2's tanh-squash,
      which can saturate the raw control when the prior is badly wrong
      and trap a later-corrected belief -- see dynamic_pricing/run_pricing_demo.py's
      REVISION HISTORY for the full diagnosis of exactly this failure.

Confirms:
  1. both chosen prices stay economically sensible -- inside their
     [p_min, p_max] bounds and move meaningfully as beliefs update, not
     pinned at either extreme for the whole horizon
  2. each seller's IDENTIFIABLE combination (b_AA-b_AB for seller A,
     b_BB-b_BA for seller B -- see PERSISTENT EXCITATION note below)
     converges to within a few percent of truth

DOES NOT CONFIRM, AND SHOULD NOT BE EXPECTED TO (see PERSISTENT
EXCITATION note below): that all four INDIVIDUAL b_hat components
converge to their true values. b_AA/b_BB (own-price terms) improve
substantially; b_AB/b_BA (cross-price terms) do NOT reliably improve, and
can end up numerically wrong (even sign-flipped) despite the controller
behaving correctly. This is flagged explicitly, not silently smoothed
over, in print_report()'s output.

PERSISTENT EXCITATION (found running this demo, 2026-08-26 -- see
CLAUDE.md's duopoly-pricing STATUS entry for the full diagnosis): once
prices settle near their current-belief profit-maximizing point, D_A =
a_A - b_AA*p_A + b_AB*p_B is observed at essentially ONE (p_A, p_B) pair
repeated every session -- textbook lack of persistent excitation. A
single repeated regressor identifies only the ONE linear combination
that determines D_A itself (here, b_AA - b_AB, since p_A ~ p_B once
settled), not the two individual coefficients. This was checked NOT to
be a solver local-optimum artifact: re-running the first solve from
several very different warm starts (oscillating, ramping-apart, random
price sequences -- see CLAUDE.md's duopoly-pricing STATUS entry)
converged to the IDENTICAL trajectory every time -- the controller isn't
stuck in a bad basin, it is correctly finding that deviating price
further to resolve the cross-term ambiguity isn't worth the immediate
profit it would cost. This is the key qualitative DIFFERENCE from the
thesis's linear toy example (applications/linear_dual_control/): there,
u2 was nearly free (1e-4 cost weight) so active probing was cheap and the
dual controller's implicit value-of-information term made it worthwhile;
here, BOTH actions are real prices with direct profit consequences, so a
rational dual controller has no similarly cheap channel to probe with,
and correctly declines to sacrifice real profit to fully resolve an
ambiguity that doesn't affect its current pricing decision.

PRICE-BOUND CHOICE: P_MAX is kept below both sellers' choke prices even
at the worst-case rival price, and P_MIN is kept high enough that the
worst-case corner (one seller's own price at P_MAX while the rival's is
at P_MIN, minimizing the offsetting cross-price boost) still leaves
demand comfortably above the dynamics.py floor clamp -- same precaution
as dynamic_pricing/run_pricing_demo.py's P_MAX note, now checked at the
2-seller worst-case corner instead of a single seller's own extreme.

Run directly: python run_duopoly_demo.py
"""

import numpy as np

from extensions.dual_control.main_outer_control_loop import main_outer_control_loop

from applications.duopoly_pricing.continuous_dynamics import continuous_dynamics
from applications.duopoly_pricing.cost import cost
from applications.duopoly_pricing.dynamics import dynamics
from applications.duopoly_pricing.duopoly_constants import DuopolyPricingConstants
from applications.duopoly_pricing.measurement import measurement

N_SESSIONS = 15
DT = 1.0

# Worst-case corner check (both sellers' bounds equal here for simplicity):
# D_A_min = a_A - b_AA_true*P_MAX + b_AB_true*P_MIN
#         = 100 - 2.0*45 + 0.3*10 = 13   (comfortably > 0)
# D_B_min = a_B - b_BB_true*P_MAX + b_BA_true*P_MIN
#         = 100 - 1.8*45 + 0.25*10 = 21.5   (comfortably > 0)
# Choke prices (D=0 at the OTHER price's floor, so this is the binding
# case): p_A_choke = (a_A + b_AB_true*P_MIN)/b_AA_true = 51.5 > P_MAX=45;
#        p_B_choke = (a_B + b_BA_true*P_MIN)/b_BB_true = 56.9 > P_MAX=45.
#
# UNCONSTRAINED JOINT-PROFIT OPTIMUM CHECK (the real constraint on P_MAX
# here, found by actually running this demo -- see CLAUDE.md's
# duopoly-pricing STATUS entry for the full diagnosis): solving
# d(profit_A+profit_B)/dp_A = d(.)/dp_B = 0 under B_TRUE gives (p_A*,
# p_B*) = (34.5, 38.2) -- both comfortably INSIDE [P_MIN, P_MAX]. This
# matters because a substitute-goods duopoly under SINGLE joint
# ownership has a real economic incentive to price ABOVE the competitive
# level (raising p_A pushes some customers to B, a positive externality
# the joint owner internalizes) -- an earlier choice of stronger
# cross-terms (0.8, 0.6) pushed this unconstrained optimum to (45.4,
# 49.2), i.e. AT OR PAST P_MAX, so the controller sat permanently pinned
# at the price ceiling even once fully informed. With prices pinned
# constant, D_A = a_A - b_AA*p + b_AB*p becomes a function of only the
# DIFFERENCE (b_AA - b_AB), not the individual terms -- confirmed
# directly: the pinned run's b_hat converged to [0.95, -0.25, -0.25,
# 0.95], sign-flipped and numerically wrong on every individual
# component, yet (b_AA_hat - b_AB_hat) = (b_BB_hat - b_BA_hat) = 1.2
# EXACTLY matched the true differences (2.0-0.8=1.2, 1.8-0.6=1.2) -- a
# genuine structural identifiability degeneracy from the bound
# permanently binding, not a solver bug. Milder cross-terms below keep
# the true optimum interior, so price is free to relax off the ceiling
# as the estimate corrects, giving the genuine session-to-session price
# variation individual-coefficient identification requires.
P_MIN, P_MAX = 10.0, 45.0

# True sensitivities [b_AA, b_AB, b_BA, b_BB] -- asymmetric on purpose
# (seller A is more own-price-sensitive than seller B, and the two
# sellers' cross-effects on each other differ) to exercise all 4
# independent parameters, not just a symmetric pair. Cross-terms are
# deliberately mild (0.3/0.25, not e.g. 0.8/0.6) -- see the P_MAX note
# above for why a stronger substitution effect pushes the true joint
# optimum outside the price bounds entirely.
B_TRUE = np.array([2.0, 0.3, 0.25, 1.8])
# Deliberately wrong initial guess -- believes both sellers are much less
# own-price-sensitive, and understates both cross-price effects too.
B_PRIOR = np.array([0.5, 0.1, 0.1, 0.5])


def run_demo(n_sessions=N_SESSIONS, seed=0, verbose=False):
    rng = np.random.default_rng(seed)  # noqa: F841 (kept for future noise-seeding hooks)

    constants = DuopolyPricingConstants(
        a_A=100.0, a_B=100.0, c_A=10.0, c_B=12.0, eta_A=0.3, eta_B=0.3,
        p_ref_A=35.0, p_ref_B=35.0, b_prior=B_PRIOR,
        F_dyn_A=1.0, F_dyn_B=1.0, F_param_A=0.05, F_param_B=0.05,
        sigma_y_A=1.0, sigma_y_B=1.0, lambda_reg_A=1e-4, lambda_reg_B=1e-4)

    # Plausible starting demand: the TRUE equilibrium demand at (p_ref_A,
    # p_ref_B) under the TRUE b -- both sellers can observe current sales
    # directly, even without knowing WHY demand responds the way it does.
    x0 = np.array([
        constants.a_A - B_TRUE[0] * constants.p_ref_A + B_TRUE[1] * constants.p_ref_B,
        constants.a_B - B_TRUE[3] * constants.p_ref_B + B_TRUE[2] * constants.p_ref_A,
    ])
    p_hat_0 = B_PRIOR.copy()
    p_true = B_TRUE.copy()
    cov_X = 1.0                       # demand-estimate variance (both sellers)
    cov_P = np.diag([1.0, 1.0, 1.0, 1.0])  # b-estimate variance -- wide
                                             # relative to |B_TRUE-B_PRIOR|
                                             # = [1.5, 0.6, 0.4, 1.3], so
                                             # the prior doesn't itself
                                             # rule out the true values

    u_lims = np.array([[P_MIN, P_MAX], [P_MIN, P_MAX]])

    result = main_outer_control_loop(
        T=n_sessions * DT, dt=DT, x_hat_0=x0, x_true_0=x0.copy(),
        p_hat_0=p_hat_0, p_true=p_true, cov_X=cov_X, cov_P=cov_P,
        constants=constants, u_lims=u_lims, u_lim_method=1,
        dynamics=dynamics, measurement=measurement, cost=cost,
        continuous_dynamics=continuous_dynamics, ny=2, nv=2,
        reg_type=1, max_du_iterations=100, first_run_max_du_iterations=150,
        augment_states_in_ilqg=True, augment_states_in_filter=True,
        verbose=verbose)

    return result, constants


def print_report(result):
    x_true, p_hat, u = result["x_true"], result["p_hat"], result["u"]
    N = x_true.shape[1]

    # With u_lim_method=1, bounds are enforced by back_pass.py's box-QP
    # directly on the control -- `u` IS the physical price already, no
    # tanh conversion needed.
    prices_A, prices_B = u[0, :], u[1, :]

    print("=== Duopoly pricing dual-control demo ===")
    print(f"True b=[b_AA,b_AB,b_BA,b_BB]={B_TRUE}")
    print(f"Initial (wrong) estimate b_hat_0={B_PRIOR}\n")
    print(f"{'Sess':>4} | {'D_A':>7} | {'D_B':>7} | {'p_A':>6} | {'p_B':>6} | "
          f"{'b_AA':>6} | {'b_AB':>6} | {'b_BA':>6} | {'b_BB':>6}")
    print("-" * 82)
    for k in range(N):
        pa_str = f"{prices_A[k]:6.2f}" if k < N - 1 else "  --  "
        pb_str = f"{prices_B[k]:6.2f}" if k < N - 1 else "  --  "
        print(f"{k:4d} | {x_true[0, k]:7.2f} | {x_true[1, k]:7.2f} | {pa_str} | "
              f"{pb_str} | {p_hat[0, k]:6.3f} | {p_hat[1, k]:6.3f} | "
              f"{p_hat[2, k]:6.3f} | {p_hat[3, k]:6.3f}")

    err0 = np.abs(p_hat[:, 0] - B_TRUE)
    errN = np.abs(p_hat[:, -1] - B_TRUE)
    names = ["b_AA", "b_AB", "b_BA", "b_BB"]

    print(f"\nIndividual b_hat |error| by component:")
    for name, e0, eN in zip(names, err0, errN):
        verdict = "improved" if eN < e0 else "DID NOT improve"
        print(f"  {name}: session 0 = {e0:6.3f}  ->  final = {eN:6.3f}  ({verdict})")

    # See run_duopoly_demo.py's module docstring, "PERSISTENT EXCITATION":
    # once prices settle, only each seller's (own - cross) combination is
    # identifiable, not the individual terms -- report that combination
    # explicitly since it's the economically meaningful, actually-checkable
    # convergence claim, not the raw components above.
    diff_A_true, diff_A_hat = B_TRUE[0] - B_TRUE[1], p_hat[0, -1] - p_hat[1, -1]
    diff_B_true, diff_B_hat = B_TRUE[3] - B_TRUE[2], p_hat[3, -1] - p_hat[2, -1]
    print(f"\nIdentifiable combinations (own - cross), the quantity that "
          f"actually determines behavior:")
    print(f"  seller A (b_AA-b_AB): true={diff_A_true:.3f}  hat={diff_A_hat:.3f}  "
          f"({100 * abs(diff_A_hat - diff_A_true) / diff_A_true:.1f}% error)")
    print(f"  seller B (b_BB-b_BA): true={diff_B_true:.3f}  hat={diff_B_hat:.3f}  "
          f"({100 * abs(diff_B_hat - diff_B_true) / diff_B_true:.1f}% error)")

    print(f"\nPrice A range chosen: [{prices_A.min():.2f}, {prices_A.max():.2f}]  "
          f"(bounds were [{P_MIN}, {P_MAX}])")
    print(f"Price B range chosen: [{prices_B.min():.2f}, {prices_B.max():.2f}]  "
          f"(bounds were [{P_MIN}, {P_MAX}])")
    print(f"Total true cost over horizon: {result['total_true_cost']:.2f}\n")

    return err0, errN, prices_A, prices_B


if __name__ == "__main__":
    result, constants = run_demo(verbose=False)
    print_report(result)
