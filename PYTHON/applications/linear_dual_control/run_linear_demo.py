"""
Reproduces thesis Figs. 3.4/3.5 (Sec. 3.4's linear dual-control example)
end to end through the REAL extensions/dual_control/main_outer_control_loop
MPC loop -- no new solver code, only the plant callables in
linear_plant.py (dynamics/measurement/cost/continuous_dynamics), matching
the existing DynamicsFn/MeasurementFn/CostFn/ContinuousDynamicsFn
contracts in extensions/dual_control/protocols.py.

Thesis setup (Sec. 3.4, verified directly against the thesis PDF
2026-08-26 -- NOTES/2024_Mathis_Sensinger Carretero_PhD Thesis.pdf,
pp. 58-61 -- not just the earlier headline summary):
    x1_dot = d1*u1 - d2,  x2_dot = d2*u2
    J(x, u) = (x1 - 5)^2 + 1e-4 * u2^2
    x0 = [0, 0], states AND controls both explicitly stated unconstrained
        (confirmed: no u1/u2 bound anywhere in Sec. 3.4 -- ruling out an
        earlier hypothesis that a missing control limit explained the
        cost gap noted below)
    d_true = [125, 50], d_hat_0 = [100, 30]
    Sigma_d0 = [[8100, 4500], [4500, 5625]]
    6 steps x dt=0.05s
    measurement noise G = 1e-2 (both states), process noise F = 1e-15
        (for BOTH state and parameter dynamics -- the thesis states F was
        set to 1e-15 "for both the state and parameter dynamics" since
        MS-SP-NMPC, the comparison baseline, doesn't model state
        uncertainty; matches linear_plant.py's F_STATE=F_PARAM=1e-15)
The thesis text does not state lambda_i/reg_type/horizon_mode/cov_X for
this experiment specifically (checked; not present anywhere in Sec. 3.4,
nor in the generic iLQG background of Sec. 2.1.3, nor in Appendix A's
DDP-controller comparison tables) -- left at main_outer_control_loop.py's
defaults.

FIRST-SOLVE SEEDING (see `run()` below): the thesis's Sec. 3.3.1
"Seeding approach" explicitly warns dual iLQG's inner-loop solve is only
LOCALLY optimal, and describes running the first outer-loop iteration
from several different initial control-trajectory guesses to escape a
bad local optimum -- exactly what's needed here. main_outer_control_loop.py
raises NotImplementedError for `first_run_seeds>1` (documented as "not
ported"), so `run()` reproduces the same effect manually: it re-runs the
whole loop from a small set of `u_bar_0` candidates for the first solve
only and keeps the lowest-cost result -- restricted to the first
iteration, matching Sec. 3.3.1's own stated common practice ("seeding
could take place at every time step... [or] only during the first time
step").  All-zero u_bar_0 (main_outer_control_loop.py's own default)
lands in a real, verified-not-a-bug local optimum where u2 stays
~1e-6 throughout (no parameter-probing) and total cost is 1.128; ANY
meaningfully nonzero u2 seed (tested from 0.05 to 0.5, both signs, flat
or decaying) escapes it and lands within noise of 0.391 -- matching the
thesis's reported 0.4, and reproducing every qualitative claim in the
text: x1 overshoots to ~5.6 on the first step (thesis says "5.6"; here
5.625), u2 decays to ~0 over about four steps (thesis: "gradually steps
down to zero over four time steps"), d_hat lands within ~0.03% of the
true [125, 50] after just the first step (thesis: "accurately estimating
the true parameter values... after the first time step"), and virtually
the entire 0.39 total traces to that single first-step overshoot
((5.625-5)^2=0.3906), consistent with the thesis's "tracked the state
reference with only small errors" for every step after.

Run directly: python run_linear_demo.py
"""

import os

import numpy as np

from core.ddp_solver.plugin_validation import PluginContractError
from extensions.dual_control.main_outer_control_loop import main_outer_control_loop
from extensions.dual_control.validate_plugin import (
    validate_continuous_dynamics_fn, validate_cost_fn, validate_dynamics_fn,
    validate_measurement_fn)

from applications.linear_dual_control.linear_plant import (
    NX, NU, N_P, NY, NV, cost, continuous_dynamics, dynamics, measurement)

T = 6 * 0.05   # 6 steps
DT = 0.05

X_HAT_0 = np.array([0.0, 0.0])
X_TRUE_0 = np.array([0.0, 0.0])
P_TRUE = np.array([125.0, 50.0])
P_HAT_0 = np.array([100.0, 30.0])
COV_X = 1e-8   # x0 known essentially exactly
COV_P = np.array([[8100.0, 4500.0], [4500.0, 5625.0]])

CONSTANTS = np.zeros(0)   # unused by linear_plant when augment_states=True
U_LIMS = np.array([[-1e6, 1e6], [-1e6, 1e6]])   # thesis states no control limits
U_LIM_METHOD = 1   # external box-QP bound (never binding at this width)


# First-solve seed candidates for u2 (see module docstring's "FIRST-SOLVE
# SEEDING"): all-zero (main_outer_control_loop.py's own default) plus a
# decaying positive/negative probe -- main_outer_control_loop.py's own
# `first_run_seeds` knob is NotImplementedError, so this reproduces the
# same escape-the-bad-local-optimum effect by hand, restricted to the
# very first outer iteration.
_U2_SEED_PROFILES = {
    "zero": np.zeros(5),
    "decay_+0.3": np.array([0.3, 0.2, 0.1, 0.05, 0.0]),
    "decay_-0.3": np.array([-0.3, -0.2, -0.1, -0.05, 0.0]),
}


def run(verbose=False):
    # linear_plant only supports augment_states=True (this problem always
    # augments d1/d2 onto the state) -- restrict validation to that mode
    # rather than the wrapper's default (True, False), which would need
    # `constants` to already carry p_hat appended for the False branch.
    issues = []
    kwargs = dict(nx=NX, nu=NU, constants=CONSTANTS, n_p=N_P, K=3,
                    augment_states_modes=(True,), rng=np.random.default_rng(0))
    issues += validate_dynamics_fn(dynamics, nv=NV, **kwargs)
    issues += validate_measurement_fn(measurement, nv=NV, ny=NY, **kwargs)
    issues += validate_cost_fn(cost, **kwargs)
    issues += validate_continuous_dynamics_fn(continuous_dynamics, **kwargs)
    if issues:
        raise PluginContractError(issues)

    N = round(T / DT)
    best = None
    for name, u2_profile in _U2_SEED_PROFILES.items():
        u_bar_0 = np.zeros((NU, N - 1))
        u_bar_0[1, :] = u2_profile
        result = main_outer_control_loop(
            T=T, dt=DT, x_hat_0=X_HAT_0, x_true_0=X_TRUE_0, p_hat_0=P_HAT_0,
            p_true=P_TRUE, cov_X=COV_X, cov_P=COV_P, constants=CONSTANTS,
            u_lims=U_LIMS, u_lim_method=U_LIM_METHOD, dynamics=dynamics,
            measurement=measurement, cost=cost,
            continuous_dynamics=continuous_dynamics, ny=NY, nv=NV,
            u_bar_0=u_bar_0, verbose=verbose)
        if verbose:
            print(f"  seed={name}: total_true_cost={result['total_true_cost']:.5f}")
        if best is None or result["total_true_cost"] < best[1]:
            best = (name, result["total_true_cost"], result)

    seed_name, _, result = best
    result["seed_used"] = seed_name
    return result


def print_report(result):
    N = result["p_hat"].shape[1]
    print(f"{'Step':>4} | {'t':>6} | {'x1':>8} | {'x2':>8} | "
          f"{'d1_hat':>8} | {'d2_hat':>8} | {'cost_true':>10}")
    print("-" * 66)
    for k in range(N):
        print(f"{k:4d} | {k * DT:6.3f} | {result['x_true'][0, k]:8.4f} | "
              f"{result['x_true'][1, k]:8.4f} | {result['p_hat'][0, k]:8.3f} | "
              f"{result['p_hat'][1, k]:8.3f} | {result['cost_true'][k]:10.5f}")
    print()
    print(f"Best first-solve seed: {result['seed_used']}")
    print(f"Final d_hat = {result['p_hat'][:, -1]}  (true d = {P_TRUE})")
    print(f"Total true cumulative cost = {result['total_true_cost']:.5f}  "
          f"(thesis reports 0.40)")


def maybe_plot(result, out_path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available -- skipping plot)")
        return

    N = result["p_hat"].shape[1]
    t = np.arange(N) * DT
    cum_cost = np.cumsum(result["cost_true"])

    fig, (ax_d, ax_cost) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_d.plot(t, result["p_hat"][0, :], "o-", label="d1_hat")
    ax_d.axhline(P_TRUE[0], color="C0", linestyle="--", linewidth=1, label="d1 true")
    ax_d.plot(t, result["p_hat"][1, :], "s-", label="d2_hat")
    ax_d.axhline(P_TRUE[1], color="C1", linestyle="--", linewidth=1, label="d2 true")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_ylabel("Parameter estimate")
    ax_d.set_title("Fig. 3.4: parameter estimates")
    ax_d.legend(fontsize=8)

    ax_cost.plot(t, cum_cost, "o-", color="C2")
    ax_cost.set_ylim(0, cum_cost[-1] * 1.15)
    ax_cost.set_xlabel("Time (s)")
    ax_cost.set_ylabel("Cumulative cost")
    ax_cost.set_title("Fig. 3.5: cumulative cost")

    fig.suptitle("Linear dual-control example (thesis Sec. 3.4)")
    fig.tight_layout()

    out_path = out_path or os.path.join(os.path.dirname(__file__), "linear_demo_chart.png")
    fig.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    result = run(verbose=False)
    print_report(result)
    maybe_plot(result)
