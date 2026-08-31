"""
End-to-end smoke-test demo: wires the reused Andrew solver core
(ilqg_function.py / simulate_system.py, imported directly -- no
duplication) together with Katie's newly-ported SCI plant/cost
(discrete_state_dynamics.py / continuous_state_dynamics.py /
measurement.py / l_cost.py) and runs the dual-iLQG controller on ONE
patient per persona, to see how a real persona's therapy_efficacy
profile and cost weights change the recovered SCIM/BBS trajectory.

WHAT'S REAL vs. PLACEHOLDER here (updated 2026-07-30, now that
define_persona.py/who_therapies.py exist):
  - REAL: the solver core, the plant/cost wiring, the recovery-ceiling/
    floor clamps, the tanh dose-squashing, the noise model -- all
    already unit-tested in isolation (see test_*.py in this directory).
  - REAL (new): persona-specific dynamics AND cost weights. Xi's
    per-therapy gains are now `who_therapies().G * persona.therapy_efficacy`
    (see define_persona.py's module docstring: THIS is the mechanism
    that makes different personas prefer different therapy mixes -- it's
    baked into the simulated physiology, not just the cost function),
    and `constants.persona` is a real `persona_weights(define_persona(name))`
    instead of the population-average fallback.
  - STILL PLACEHOLDER: Xi itself is still a DIRECT construction
    (population gain x persona efficacy), not a SINDy-IDENTIFIED model.
    That requires generate_synthetic_dataset.m's synthetic-patient
    generation (not yet ported) feeding build_sindy_library.py through
    prune_sindy_bic.py for that persona's own data -- a further step
    beyond what this demo does. augment_states=True with n_p=0 still
    sidesteps ilqg_function.py's augment_states=False branch, which
    assumes `constants` is a plain ndarray -- documented constraint of
    that branch, not a workaround for a bug.

Run directly: python run_katie_demo.py -- runs the SAME admission state
and horizon through TWO personas ("default" -- reproduces the original
population-average-only demo exactly, since define_persona('default')'s
weights match l_cost.py's own fallback and its therapy_efficacy is all
1.0 -- and one illustrative real persona) so the trajectories can be
compared directly.
"""

import os

import numpy as np

from extensions.dual_control.ilqg_function import ilqg_function
from extensions.dual_control.simulate_system import simulate_system

from applications.sci_rehab.continuous_state_dynamics import continuous_state_dynamics  # noqa: E402 (unused directly, imported by discrete_state_dynamics)
from applications.sci_rehab.define_persona import define_persona, persona_weights
from applications.sci_rehab.discrete_state_dynamics import discrete_state_dynamics
from applications.sci_rehab.l_cost import l_cost
from applications.sci_rehab.measurement import measurement
from applications.sci_rehab.sindy_model_constants import SindyModelConstants
from applications.sci_rehab.who_therapies import who_therapies

NX = 6   # SCIM, BBS, AIS, Age, DPI, Caregiver
NU = 13  # WHO Package of Interventions therapy doses, U1..U13
N_TERMS = 1 + (NX + NU)  # bias + linear-only library (poly_order=1)

# who_therapies.py's population-average gains sum to 2.49 (SCIM) / 1.51
# (BBS) if all 13 therapies were dosed at max simultaneously in one
# week -- more than enough to jump straight to the recovery ceiling in
# a single session, since nothing in this linear placeholder models
# diminishing returns across a therapy MIX (that's exactly the kind of
# structure real SINDy identification, and BIC/sparsity pruning, would
# be expected to find -- a handful of dominant sparse terms, not all 13
# acting at full, undiminished strength at once). Scaled down so the
# demo shows week-over-week progress instead of saturating in one step;
# purely a placeholder-plant concession, not a claim about real gains.
DEMO_GAIN_SCALE = 0.08

DEMO_PERSONAS = ["default", "frail_elder"]


def build_persona_constants(persona_name, swat_ceiling=0.85):
    persona = define_persona(persona_name)
    T = who_therapies()

    # Persona-specific gains: population-average gain (who_therapies.py)
    # x this persona's per-therapy efficacy multiplier -- see
    # define_persona.py's module docstring for why this (not the cost
    # function) is what makes different personas prefer different
    # therapy mixes.
    G_scim_persona = DEMO_GAIN_SCALE * T.G[0, :] * persona.therapy_efficacy
    G_bbs_persona = DEMO_GAIN_SCALE * T.G[1, :] * persona.therapy_efficacy

    Xi = np.zeros((N_TERMS, NX))
    u_terms = slice(1 + NX, 1 + NX + NU)  # linear u1..u13 coefficient rows
    Xi[u_terms, 0] = G_scim_persona  # dSCIM/dt
    Xi[u_terms, 1] = G_bbs_persona   # dBBS/dt

    constants = SindyModelConstants(
        Xi=Xi,
        lib_cfg={"poly_order": 1, "include_cross": False},
        swat_ceiling=swat_ceiling,
        xi_idx=np.zeros((0, 2), dtype=int),  # n_p=0: no estimated parameters
        F_dyn=1e-3, F_param=1e-4,
        persona=persona_weights(persona))

    return constants, persona


def run_demo(persona_name, n_sessions=12, verbose=False):
    constants, persona = build_persona_constants(persona_name)

    # Admission state: low SCIM/BBS, fixed patient attributes normalised
    # to [0,1] (AIS, Age, DPI, Caregiver support are placeholders here --
    # generate_synthetic_dataset.m will sample these from the persona's
    # own ais_probs/age_range/dpi_range/caregiver_alpha,beta once ported).
    x0 = np.array([0.20, 0.15, 0.5, 0.4, 0.3, 0.7])
    p_hat = np.zeros(0)  # no augmented (estimated) SINDy parameters
    cov_xa_hat_0 = 1e-3 * np.eye(NX)

    N = n_sessions
    dt = 1.0  # 1 week per session
    l0 = np.zeros((NU, N))
    L0 = np.zeros((NU, NX, N))
    u_bar0 = np.zeros((NU, N))
    u_lims = np.tile(np.array([[0.0, 1.0]]), (NU, 1))  # dose in [0,1] per therapy

    xa0, u_bar, l, L, lam, dlambda, cost_arr, Pw, Pv, converged = ilqg_function(
        T=N * dt, dt=dt, x0=x0, l=l0, L=L0, u_bar=u_bar0, lam=1.0, dlambda=1.0,
        constants=constants, p_hat=p_hat, cov_xa_hat_0=cov_xa_hat_0,
        augment_states=True, reg_type=1, u_lims=u_lims, ny=2, nv=2, nw=NX,
        max_du_iterations=100, dyn_noise_reg=0.0, tracking_trajectory=None,
        u_lim_method=2, dynamics=discrete_state_dynamics, measurement=measurement,
        cost=l_cost, simulate_system_fn=simulate_system, verbose=verbose)

    return dict(xa=xa0, u_bar=u_bar, cost=cost_arr, converged=converged,
                 constants=constants, persona=persona, N=N)


def print_report(result):
    xa, u_bar, converged, N = result["xa"], result["u_bar"], result["converged"], result["N"]
    persona = result["persona"]
    swat = result["constants"].swat_ceiling

    converged_msg = {0: "did not converge", 1: "gradient tolerance",
                       2: "cost tolerance", 3: "max iterations",
                       -1: "lambda exceeded max"}

    print(f"=== Persona: {persona.name} ===")
    print(f"  {persona.description}")
    print(f"  w_scim={persona.w_scim}  w_bbs={persona.w_bbs}  "
          f"w_resource={persona.w_resource}  "
          f"w_terminal=({persona.w_terminal_scim}, {persona.w_terminal_bbs})")
    print(f"Converged: {converged_msg.get(converged, converged)}  "
          f"(recovery ceiling swat={swat})\n")

    # Physical (squashed) dose per session, matching what the plant/cost
    # actually see -- u_lims=[0,1], u_lim_method=2's squash formula.
    u_phys = 0.5 * np.tanh(u_bar) + 0.5

    print(f"{'Session':>7} | {'SCIM':>6} | {'BBS':>6} | {'mean dose':>9}")
    print("-" * 38)
    for k in range(N + 1):
        dose_str = f"{u_phys[:, k].mean():9.3f}" if k < N else " " * 9
        print(f"{k:7d} | {xa[0, k]:6.3f} | {xa[1, k]:6.3f} | {dose_str}")

    mean_dose_per_therapy = u_phys.mean(axis=1)
    top3 = np.argsort(mean_dose_per_therapy)[::-1][:3]
    print("\nTop-3 therapies by mean dose (U-index, mean dose):")
    for idx in top3:
        print(f"  U{idx + 1}: {mean_dose_per_therapy[idx]:.3f}")
    print()


def maybe_plot(results, out_path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available -- skipping plot)")
        return

    fig, (ax_scim, ax_bbs) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for result in results:
        xa, N = result["xa"], result["N"]
        persona_name = result["persona"].name
        sessions = np.arange(N + 1)
        ax_scim.plot(sessions, xa[0, :], "o-", label=persona_name)
        ax_bbs.plot(sessions, xa[1, :], "o-", label=persona_name)

    swat = results[0]["constants"].swat_ceiling
    for ax, title in [(ax_scim, "SCIM"), (ax_bbs, "BBS")]:
        ax.axhline(swat, color="gray", linestyle="--", linewidth=1, label="recovery ceiling")
        ax.set_xlabel("Session (week)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    ax_scim.set_ylabel("Normalised score")
    fig.suptitle("Katie demo: SCIM/BBS trajectory by persona")
    fig.tight_layout()

    out_path = out_path or os.path.join(os.path.dirname(__file__), "demo_trajectory.png")
    fig.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    results = []
    for persona_name in DEMO_PERSONAS:
        result = run_demo(persona_name, n_sessions=12, verbose=False)
        print_report(result)
        results.append(result)

    maybe_plot(results)
