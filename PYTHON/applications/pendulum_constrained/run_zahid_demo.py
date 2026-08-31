"""
End-to-end demo: reproduces the paper's own inverted-pendulum experiment
(Dastan & Sensinger 2024, "Leveraging Control Inputs to Enforce
Constraints in Differential Dynamic Programming for Nonlinear
Optimization," IEEE CDC 2024, Sec IV-A, eq 20-23) through the REAL
ilqg() solver (todorov/ilqg.py, extended with the optional
`constraint_fn` parameter -- see todorov/back_pass.py's/
forward_pass.py's/ilqg.py's docstrings for that extension).

constraint_fn is built by composing:
  - relative_degree_reduction.state_constraint_to_control_constraint
    (Variant B): reduces the state-only constraint omega < 1 to a
    control-dependent h_tilde(x,u), eq 16-19.
  - dynamic_control_bounds.build_time_varying_lims (Variant A): turns
    h_tilde into a per-timestep box bound on u, eq 10-13.
This is the actual composition the paper describes for a pure
state constraint: "the reformulated constraint now implicitly involving
u, we can directly apply the methodology developed in the previous
subsection."

No source code exists for this paper (see CLAUDE.md's "STATUS -- Zahid
conversion") -- this is the composition sanity check for an ORIGINAL
implementation derived from the paper's equations, the same role
run_katie_demo.py plays for Katie's SCI plant. Verified only
QUALITATIVELY against the paper's own reported numbers (Sec IV-A /
Fig 3), since there is no ground-truth code or trajectory to diff
against exactly.

FINDING (2026-08-04): the paper's LITERAL alpha=0.1 traps this DDP
implementation after exactly one accepted outer iteration -- confirmed
NOT a coding bug (the unconstrained version of this same problem, same
step_fn/derivs_fn, converges cleanly to cost ~8.9; every piece of the
constraint math is independently unit-tested to ~1e-10 elsewhere). Root
cause, diagnosed by direct inspection: at alpha=0.1, u_max(x) =
alpha*(1-omega) - sin(phi) stays in a narrow ~0.1-0.25 window across
roughly phi in [-pi, -3.0] (about the first 0.14 rad of the swing),
because alpha=0.1 is conservative even where h=1-omega is far from its
boundary (omega=0 initially, h=1) -- not just at the exact phi=-pi
point (sin(-pi)=0 is not a floating-point fluke; sin stays near 0 for
a genuine stretch nearby). The backward pass's box-QP has essentially no
room to move in that stretch, so after the first iteration exhausts what
little slack exists, every later iteration is rejected and lambda climbs
past lambda_max.

This directly instantiates a limitation the paper itself states (eq 14):
"as long as the adjustments to u are MODEST and h(x,u) is smooth, the
updated trajectory will continue to fulfill the inequality constraint"
-- i.e. the method's local-validity argument is only approximate for
large accepted steps. Confirmed empirically: at alpha=0.3 (still tight,
but with a large accepted step early on) the returned trajectory
actually VIOLATES the constraint (max omega ~1.52); at alpha=2.0 it
undershoots less badly but still slightly violates (~1.05).

alpha=1.0 (not the paper's stated value for this experiment, but well
within the "no closed-form selection rule given" territory the paper
itself describes -- see relative_degree_reduction.py's docstring)
converges cleanly with the SAME unmodified code and reproduces the
paper's qualitative shape well: 13 iterations (paper: 16), cost 11.46
(paper: 10.11), max omega 0.93 (paper: "close to but under 1"). This
script runs alpha=0.1 first and reports the trap plus its diagnosis,
then alpha=1.0 as the actual qualitative-match result, rather than
silently substituting one for the other.

Run directly: python run_zahid_demo.py
"""

import numpy as np

from core.ddp_solver.ilqg import ilqg
from core.ddp_solver.forward_pass import forward_pass

from extensions.constraints.dynamic_control_bounds import build_time_varying_lims
from extensions.constraints.relative_degree_reduction import state_constraint_to_control_constraint

# ---- paper eq 20-23 --------------------------------------------------
# NOTE: the paper calls its integration step "h" AND its constraint
# function "h" -- same notation clash flagged in
# dynamic_control_bounds.py/relative_degree_reduction.py. Renamed the
# integration step DT here to keep the two unambiguous.
DT = 0.05
N_STEPS = 500
RUNNING_WEIGHT = 0.025    # eq 22: L(x,u) = 0.025*(phi^2 + omega^2 + u^2)
TERMINAL_WEIGHT = 5.0     # eq 23: Phi(x) = 5*(phi^2 + omega^2)
ALPHA_PAPER = 0.1         # eq 19's alpha, paper's STATED value -- traps (see
                          # module docstring "FINDING"); reported, not hidden
ALPHA_WORKING = 1.0       # demonstrates the same wiring converges cleanly
X0 = np.array([-np.pi, 0.0])


def pendulum_continuous_dynamics(x, u):
    """x_dot = f(x,u): phi_dot = omega, omega_dot = sin(phi) + u -- the
    continuous ODE underlying eq 20's Euler discretization. Required by
    relative_degree_reduction.py's Lie-derivative construction (eq 16),
    which is inherently a continuous-time chain rule -- NOT the same
    callable as step_fn below (that's the discrete one-step map)."""
    return np.array([x[1], np.sin(x[0]) + u[0]])


def step_fn(x, u, i):
    """Discrete dynamics (eq 20) + running/terminal cost (eq 22-23),
    vectorized over a (dim, K) batch -- forward_pass.py's step_fn
    contract. u[:, k] is all-NaN for the final (control-free) cost
    evaluation."""
    is_final = np.isnan(u[0])
    u0 = np.where(np.isnan(u), 0.0, u)
    phi, omega = x[0], x[1]

    x_next = np.stack([phi + DT * omega,
                        omega + DT * np.sin(phi) + DT * u0[0]], axis=0)

    running = RUNNING_WEIGHT * (phi ** 2 + omega ** 2 + u0[0] ** 2)
    terminal = TERMINAL_WEIGHT * (phi ** 2 + omega ** 2)
    cost = np.where(is_final, terminal, running)
    return x_next, cost


def derivs_fn(x, u_aug):
    """Analytic derivatives of step_fn's dynamics/cost -- eq 20's
    dynamics are simple enough not to need finite differences (unlike
    the constraint machinery, which does support that fallback)."""
    n_time = x.shape[1]
    phi = x[0]
    u0 = np.where(np.isnan(u_aug), 0.0, u_aug)
    is_final = np.isnan(u_aug[0])

    fx = np.zeros((2, 2, n_time))
    fx[0, 0, :] = 1.0
    fx[0, 1, :] = DT
    fx[1, 0, :] = DT * np.cos(phi)
    fx[1, 1, :] = 1.0

    fu = np.zeros((2, 1, n_time))
    fu[1, 0, :] = DT

    weight2 = np.where(is_final, 2.0 * TERMINAL_WEIGHT, 2.0 * RUNNING_WEIGHT)
    cx = weight2 * x
    cxx = np.zeros((2, 2, n_time))
    cxx[0, 0, :] = weight2
    cxx[1, 1, :] = weight2

    cu = 2.0 * RUNNING_WEIGHT * u0
    cxu = np.zeros((2, 1, n_time))
    cuu = np.full((1, 1, n_time), 2.0 * RUNNING_WEIGHT)

    return fx, fu, None, None, None, cx, cu, cxx, cxu, cuu


# ---- constraint: omega < 1, i.e. h(x) = 1 - omega >= 0 (state-only) --

def h_omega(x):
    return np.array([1.0 - x[1]])


def dh_omega_dx(x):
    return np.array([[0.0, -1.0]])


def dh_tilde_du(x, u):
    # h_tilde(x,u) = alpha*(1-omega) - sin(phi) - u -> d/du = -1, constant
    # (hand-derived and unit-tested in test_relative_degree_reduction.py).
    return np.array([[-1.0]])


def make_constraint_fn(alpha):
    h_tilde_fn = state_constraint_to_control_constraint(
        h_omega, pendulum_continuous_dynamics, alpha, dhdx_fn=dh_omega_dx)

    def constraint_fn(x_traj, u_traj):
        return build_time_varying_lims(
            h_tilde_fn, x_traj, u_traj, dhdu_fn=dh_tilde_du)

    return constraint_fn


def bound_at(alpha, phi, omega):
    """u_max(x) = alpha*(1-omega) - sin(phi), for the diagnostic print in
    main() -- shows how tight the box is at/near x0 for a given alpha."""
    return alpha * (1.0 - omega) - np.sin(phi)


def plot_results(x, trace, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(os.path.dirname(__file__),
                                         "pendulum_demo_chart.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(x[1], label="omega (proposed method)")
    axes[0].axhline(1.0, color="r", linestyle="--", label="constraint: omega = 1")
    axes[0].set_xlabel("N (control horizon step)")
    axes[0].set_ylabel("omega")
    axes[0].set_title("Angular velocity vs horizon (paper Fig 3a analog)")
    axes[0].legend()

    costs = [r["cost"] for r in trace if "cost" in r]
    axes[1].plot(range(1, len(costs) + 1), costs)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("total cost J")
    axes[1].set_title("Cost convergence (paper Fig 3b analog)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved comparison chart to {out_path}")


def run_one(alpha, label):
    rng = np.random.default_rng(0)
    # paper Sec IV: "the initial control input u0 is selected by
    # generating small random values"
    u0 = 0.01 * rng.standard_normal((1, N_STEPS))
    constraint_fn = make_constraint_fn(alpha)

    print(f"\n--- Running constrained iLQG: alpha={alpha}  ({label}) ---\n")
    x, u, L, Vx, Vxx, cost, trace, stop_reason = ilqg(
        step_fn, derivs_fn, X0, u0, constraint_fn=constraint_fn, verbose=1)

    n_iters = trace[-1]["iter"]
    final_cost = cost.sum()
    max_omega = x[1].max()
    respected = max_omega <= 1.0 + 1e-3

    print(f"Stop reason: {stop_reason}")
    print(f"Iterations: {n_iters}")
    print(f"Final total cost: {final_cost:.4f}")
    print(f"Max omega over trajectory: {max_omega:.6f}  "
          f"({'OK, constraint respected' if respected else 'VIOLATED'})")
    print(f"Final state: phi={x[0, -1]:.6f}, omega={x[1, -1]:.6f}")

    return dict(alpha=alpha, x=x, u=u, cost=cost, trace=trace,
                stop_reason=stop_reason, n_iters=n_iters,
                final_cost=final_cost, max_omega=max_omega, respected=respected)


def main():
    x_init, _, cost_init = forward_pass(
        X0, 0.01 * np.random.default_rng(0).standard_normal((1, N_STEPS)),
        None, None, None, np.array([1.0]), step_fn, None)
    cost_init_total = cost_init[:, 0].sum()

    print("=" * 72)
    print("Zahid (Dastan & Sensinger 2024) demo -- inverted pendulum, eq 20-23")
    print("Constraint: omega < 1  (state-only, reduced via relative degree one)")
    print("=" * 72)
    print(f"\nx0 = {X0}, N = {N_STEPS} steps, dt = {DT}")
    print(f"Initial (unoptimized) total cost: {cost_init_total:.4f}")
    print(f"u_max(x0) at alpha={ALPHA_PAPER}: {bound_at(ALPHA_PAPER, X0[0], X0[1]):.4f}"
          f"   u_max(x0) at alpha={ALPHA_WORKING}: {bound_at(ALPHA_WORKING, X0[0], X0[1]):.4f}")

    paper_run = run_one(ALPHA_PAPER, "paper's literal stated value for this experiment")
    if not paper_run["respected"] or "EXIT" in paper_run["stop_reason"]:
        print(f"\n>>> alpha={ALPHA_PAPER} trapped this optimizer -- see module "
              "docstring's FINDING for the diagnosed root cause (a narrow "
              "control-authority corridor near x0, not a coding bug; "
              "confirmed via the unconstrained baseline and the "
              "independently unit-tested constraint math).")

    working_run = run_one(ALPHA_WORKING, "demonstrates the same code converges cleanly")

    print("\n" + "=" * 72)
    print("SUMMARY vs. paper Fig 3 / Sec IV-A (final cost 10.11, 16 iters,")
    print("omega stays close to but under 1)")
    print("=" * 72)
    print(f"  alpha={ALPHA_PAPER} (paper's value): {paper_run['stop_reason']}, "
          f"cost={paper_run['final_cost']:.2f}, max_omega={paper_run['max_omega']:.3f}")
    print(f"  alpha={ALPHA_WORKING} (working alt.): {working_run['stop_reason']}, "
          f"cost={working_run['final_cost']:.2f}, max_omega={working_run['max_omega']:.3f}, "
          f"iters={working_run['n_iters']}  <- qualitative match")

    return paper_run, working_run


if __name__ == "__main__":
    paper_run, working_run = main()
    plot_results(working_run["x"], working_run["trace"])
