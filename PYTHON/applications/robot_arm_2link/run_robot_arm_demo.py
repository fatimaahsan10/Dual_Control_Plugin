"""
End-to-end dual-control demo for the 2-link robot arm plug-in: wires the
reused Andrew/Katie solver core (main_outer_control_loop.py, imported
directly -- no duplication) to the arm plant
(dynamics.py/continuous_dynamics.py/measurement.py/cost.py), starting
from a DELIBERATELY WRONG initial guess for link 2's mass while the arm
is torque-controlled to reach a fixed joint-space target.

Applies both hard-won lessons already written up in this codebase (see
applications/dynamic_pricing/run_pricing_demo.py's REVISION HISTORY and
applications/duopoly_pricing/run_duopoly_demo.py's docstring) from the
start: u_lim_method=1 (box-QP bound enforcement in back_pass.py,
operating on torque directly), not u_lim_method=2's tanh-squash.

TORQUE-BOUND CHOICE: the largest gravity torque the TRUE arm (m2=1.0 kg)
ever needs to hold statically is at q1=q2=0 (arm horizontal, both links
fully extended -- the worst lever arm): G1 = (m1*lc1 + m2*l1)*g +
m2*lc2*g = (0.5 + 1.0)*9.81 + 0.5*9.81 = 19.62 N*m, G2 = m2*lc2*g = 4.905
N*m. TAU_LIMS below (+-30 / +-15 N*m) leave roughly 10 N*m of headroom on
each joint beyond the static gravity hold for actually accelerating the
arm toward the target -- checked directly against this worst-case
config, not just the target config (which needs less: ~12.1 / ~4.7 N*m).

PERSISTENT EXCITATION EXPECTATION (see the math-review discussion this
plug-in was proposed against, CLAUDE.md): unlike
applications/duopoly_pricing/'s price, which settles to a fixed point
and STOPS exciting the unknown once profit-optimal, reaching q1_target/
q2_target from rest structurally REQUIRES the arm to move through a
range of q2 (and therefore accelerate against m2-dependent inertia/
Coriolis/gravity terms) before it can settle -- so m2 should be
identifiable from the transient alone, without needing a dedicated
probing incentive the way duopoly's cross-price terms did. Whether this
actually holds is checked here empirically (print_report reports the
real error, not an assumed one), same "run it, don't just assert it"
standard as every other demo in this codebase.

Run directly: python run_robot_arm_demo.py
"""

import numpy as np

from extensions.dual_control.main_outer_control_loop import main_outer_control_loop

from applications.robot_arm_2link.arm_constants import RobotArmConstants
from applications.robot_arm_2link.continuous_dynamics import continuous_dynamics
from applications.robot_arm_2link.cost import cost
from applications.robot_arm_2link.dynamics import dynamics
from applications.robot_arm_2link.measurement import measurement

N_SESSIONS = 40
DT = 0.05  # s -- 2.0 s total reach horizon

TAU1_LIM, TAU2_LIM = 30.0, 15.0  # N*m -- see module docstring's worst-case check

# True vs. deliberately-wrong-prior link-2 mass.
M2_TRUE = 1.0    # kg
M2_PRIOR = 0.3   # kg -- badly understates link 2's mass/inertia contribution

Q1_TARGET, Q2_TARGET = np.pi / 3, -np.pi / 4  # 60 deg, -45 deg


def run_demo(n_sessions=N_SESSIONS, seed=0, verbose=False):
    rng = np.random.default_rng(seed)  # noqa: F841 (kept for future noise-seeding hooks)

    constants = RobotArmConstants(
        l1=1.0, l2=1.0, lc1=0.5, lc2=0.5, m1=1.0, I1=0.1, I2=0.1, g=9.81,
        fv1=0.5, fv2=0.5, m2_prior=M2_PRIOR, q1_target=Q1_TARGET, q2_target=Q2_TARGET)

    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # arm at rest, horizontal
    p_hat_0 = np.array([M2_PRIOR])
    p_true = np.array([M2_TRUE])
    cov_X = 0.01                    # state-estimate variance (rad, rad/s)
    cov_P = np.array([[0.25]])      # m2-estimate variance -- std 0.5 kg,
                                      # wide relative to |M2_TRUE-M2_PRIOR|=0.7

    u_lims = np.array([[-TAU1_LIM, TAU1_LIM], [-TAU2_LIM, TAU2_LIM]])

    result = main_outer_control_loop(
        T=n_sessions * DT, dt=DT, x_hat_0=x0, x_true_0=x0.copy(),
        p_hat_0=p_hat_0, p_true=p_true, cov_X=cov_X, cov_P=cov_P,
        constants=constants, u_lims=u_lims, u_lim_method=1,
        dynamics=dynamics, measurement=measurement, cost=cost,
        continuous_dynamics=continuous_dynamics, ny=4, nv=4,
        reg_type=1, max_du_iterations=100, first_run_max_du_iterations=150,
        augment_states_in_ilqg=True, augment_states_in_filter=True,
        verbose=verbose)

    return result, constants


def print_report(result):
    x_true, p_hat, u = result["x_true"], result["p_hat"], result["u"]
    N = x_true.shape[1]

    # With u_lim_method=1, bounds are enforced by back_pass.py's box-QP
    # directly on the control -- `u` IS the physical torque already, no
    # tanh conversion needed.
    tau1, tau2 = u[0, :], u[1, :]

    print("=== 2-link robot arm dual-control demo ===")
    print(f"True m2={M2_TRUE} kg   Initial (wrong) estimate m2_hat_0={M2_PRIOR} kg")
    print(f"Target: q1={np.degrees(Q1_TARGET):.1f} deg, q2={np.degrees(Q2_TARGET):.1f} deg\n")
    print(f"{'Sess':>4} | {'q1(deg)':>8} | {'q2(deg)':>8} | {'w1':>6} | {'w2':>6} | "
          f"{'tau1':>7} | {'tau2':>7} | {'m2_hat':>7}")
    print("-" * 78)
    for k in range(N):
        t1_str = f"{tau1[k]:7.2f}" if k < N - 1 else "   --  "
        t2_str = f"{tau2[k]:7.2f}" if k < N - 1 else "   --  "
        print(f"{k:4d} | {np.degrees(x_true[0, k]):8.2f} | {np.degrees(x_true[1, k]):8.2f} | "
              f"{x_true[2, k]:6.2f} | {x_true[3, k]:6.2f} | {t1_str} | {t2_str} | "
              f"{p_hat[0, k]:7.3f}")

    err0 = abs(p_hat[0, 0] - M2_TRUE)
    errN = abs(p_hat[0, -1] - M2_TRUE)
    verdict = "improved" if errN < err0 else "DID NOT improve"
    print(f"\nm2_hat |error|: session 0 = {err0:.3f}  ->  final = {errN:.3f}  ({verdict})")

    final_q1, final_q2 = x_true[0, -1], x_true[1, -1]
    print(f"\nFinal q1={np.degrees(final_q1):.2f} deg (target {np.degrees(Q1_TARGET):.2f}), "
          f"error={np.degrees(final_q1 - Q1_TARGET):.2f} deg")
    print(f"Final q2={np.degrees(final_q2):.2f} deg (target {np.degrees(Q2_TARGET):.2f}), "
          f"error={np.degrees(final_q2 - Q2_TARGET):.2f} deg")
    print(f"Final w1={x_true[2, -1]:.3f} rad/s, w2={x_true[3, -1]:.3f} rad/s (settling check)")

    print(f"\nTorque 1 range chosen: [{tau1.min():.2f}, {tau1.max():.2f}]  "
          f"(bounds were [{-TAU1_LIM}, {TAU1_LIM}])")
    print(f"Torque 2 range chosen: [{tau2.min():.2f}, {tau2.max():.2f}]  "
          f"(bounds were [{-TAU2_LIM}, {TAU2_LIM}])")
    print(f"Total true cost over horizon: {result['total_true_cost']:.2f}\n")

    return err0, errN, final_q1, final_q2


if __name__ == "__main__":
    result, constants = run_demo(verbose=False)
    print_report(result)
