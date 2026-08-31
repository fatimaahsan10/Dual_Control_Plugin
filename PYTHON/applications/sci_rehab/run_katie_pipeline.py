"""
Top-level, end-to-end Katie pipeline: generate synthetic data for a
chosen persona -> identify a sparse SINDy model -> validate it ->
compute its Fisher information -> report it in clinician-readable form
-> run DUAL-CONTROL therapy planning with the IDENTIFIED model,
online-refining belief about a few clinically decision-relevant
coefficients as therapy is delivered.

This is the final integration piece: every component below already
exists and is independently unit-tested (see test_*.py in this
directory); this script is the first place they all run together,
start to end, the way main_sindy_pipeline.m + Main_Outer_Control_Loop.m
would in the MATLAB source.

STRUCTURE, adapted from Main_Outer_Control_Loop.m:
  PART A (identification, mirrors main_sindy_pipeline.m's step order):
    generate_synthetic_dataset -> build_sindy_library ->
    cross_validate_lambda -> stlsq -> prune_sindy_bic -> zero static
    state equations -> validate_sindy_model -> compute_fisher_information
    -> print_identified_equations
  PART B (dual control): reuses andrew/main_outer_control_loop.py's
    receding-horizon MPC structure UNMODIFIED -- that file already
    takes dynamics/measurement/cost/continuous_dynamics as plant-
    agnostic callables (built that way for exactly this kind of reuse,
    same pattern already used for ilqg_function/simulate_system
    throughout this port) -- so this script just points it at Katie's
    SCI plant (discrete_state_dynamics.py/measurement.py/l_cost.py/
    continuous_state_dynamics.py) with the IDENTIFIED Xi, instead of
    duplicating that file's re-plan/step/re-estimate/warm-start loop.

WHICH COEFFICIENTS ARE TREATED AS "UNCERTAIN" (dual-estimated online):
  for each dynamic state (SCIM, BBS), the top-K (by |identified
  coefficient|) nonzero PURE LINEAR ACTION TERMS (a1..a13) -- exactly
  the pool build_sindy_library.py's protected_idx already prioritises
  keeping through STLSQ/BIC pruning, and the most clinically decision-
  relevant unknowns ("how much does THIS therapy actually help THIS
  state"). Their initial covariance cov_P is seeded from
  compute_fisher_information.py's Sigma (the Cramer-Rao bound diagonal
  for those library terms), de-normalised into physical-coefficient
  units via the same col_std/dX_std convention stlsq.py/
  prune_sindy_bic.py already use -- this is the "connective tissue"
  role compute_fisher_information.py's own docstring describes, wired
  up end to end here for the first time.

THE "TRUE VS ESTIMATED" STORY, mirroring Andrew's own validated dual-
control demo (see CLAUDE.md's Andrew STATUS: "deliberately wrong
parameter (0.4) correctly converges toward true value (1.0)"): p_hat_0
is seeded from the SINDy-IDENTIFIED coefficients (not artificially
perturbed -- there's no need to fake a wrong guess when real
identification noise already supplies one), and p_true comes from
generate_synthetic_dataset.py's own Xi_true (the data generator's
approximate ground truth) at the SAME coefficients. The TRUE simulated
patient in this demo uses the IDENTIFIED model for every OTHER
coefficient -- only the few marked-uncertain ones are swapped for their
true value -- so this validates dual control's ability to refine a
SMALL, targeted set of clinically-critical unknowns, not a full online
re-identification of the whole model.

Run directly: python run_katie_pipeline.py
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from extensions.dual_control.main_outer_control_loop import main_outer_control_loop

from extensions.sindy.build_sindy_library import build_sindy_library
from extensions.sindy.compute_fisher_information import compute_fisher_information
from applications.sci_rehab.continuous_state_dynamics import continuous_state_dynamics  # noqa: E402 (unused directly, imported by discrete_state_dynamics)
from extensions.sindy.cross_validate_lambda import cross_validate_lambda
from applications.sci_rehab.define_persona import Persona, define_persona, persona_weights
from applications.sci_rehab.discrete_state_dynamics import discrete_state_dynamics
from applications.sci_rehab.generate_synthetic_dataset import SyntheticDataset, generate_synthetic_dataset
from applications.sci_rehab.l_cost import l_cost
from applications.sci_rehab.measurement import measurement
from extensions.sindy.print_identified_equations import print_identified_equations
from extensions.sindy.prune_sindy_bic import prune_sindy_bic
from applications.sci_rehab.sindy_model_constants import SindyModelConstants
from extensions.sindy.stlsq import stlsq
from extensions.sindy.validate_sindy_model import validate_sindy_model

NX = 6
NU = 13
LIB_CFG = {"poly_order": 2, "include_cross": True, "include_trig": False}


@dataclass
class IdentificationResult:
    persona: Persona
    dataset: SyntheticDataset
    Theta: np.ndarray
    dX: np.ndarray
    lib_labels: List[str]
    protected_idx: np.ndarray
    lambda_opt: float
    Xi: np.ndarray
    nmae: float
    rmse: float
    F: np.ndarray
    Sigma: np.ndarray
    active_idx: np.ndarray


def run_identification(persona_name, n_patients=60, n_sessions=12, noise_std=0.03,
                          val_fraction=0.2, rng=None, verbose=True):
    if rng is None:
        rng = np.random.default_rng()

    persona = define_persona(persona_name)
    gen_cfg = {"n_patients": n_patients, "n_sessions": n_sessions,
                 "noise_std": noise_std, "persona": persona}
    dataset = generate_synthetic_dataset(gen_cfg, rng=rng)

    n_val = max(1, round(val_fraction * n_patients))
    perm = rng.permutation(n_patients)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    Theta, dX, lib_labels, protected_idx = build_sindy_library(
        dataset.X[:, :, train_idx], dataset.A[:, :, train_idx], LIB_CFG)

    # cv_cfg deliberately matches the settings already proven fast across
    # every other test_*.py composition check in this directory (n_lambda=8,
    # max_iter=20) rather than main_sindy_pipeline.m's larger defaults
    # (n_lambda=20) -- with n_terms=210 (poly_order=2+cross at 6/13 dims),
    # 5-fold CV runs stlsq() 5*n_lambda+1 times, so this setting matters a
    # lot for keeping an end-to-end demo script fast.
    cv_cfg = {"n_lambda": 8, "max_iter": 20, "tol": 1e-4,
               "max_terms": 22, "min_terms": 3}
    lambda_opt = cross_validate_lambda(
        Theta, dX, cv_cfg, protected_idx=protected_idx,
        random_state=int(rng.integers(0, 2**31 - 1)))

    Xi = stlsq(Theta, dX, lambda_opt, cv_cfg["max_iter"], cv_cfg["tol"], False,
                cv_cfg["max_terms"], protected_idx, cv_cfg["min_terms"])
    Xi, _bic_history = prune_sindy_bic(
        Xi, Theta, dX, lib_labels, dataset.state_names, False, protected_idx)
    Xi[:, 2:] = 0.0  # zero static state equations (main_sindy_pipeline.m step 4c)

    nmae, rmse, _sim_trajectories = validate_sindy_model(
        Xi, dataset.X[:, :, val_idx], dataset.A[:, :, val_idx],
        dataset.Y[:, :, val_idx], lib_labels, LIB_CFG)

    F = compute_fisher_information(Theta, noise_std)

    # Sigma (used to seed the dual-control filter's initial parameter
    # covariance, see seed_cov_P) is deliberately computed from the
    # ACTIVE terms only (nonzero in Xi), not the full 210-column library.
    # The full library's Theta.T@Theta is severely rank-deficient
    # (structural collinearity among unused polynomial/cross terms --
    # cond ~1e14-1e15 is typical, see compute_fisher_information.py's own
    # docstring), which otherwise seeds cov_P with variances in the
    # 1e5-1e6 range for coefficients whose true magnitude is ~1e-2,
    # causing the online dual-control estimator to diverge (empirically
    # confirmed via run_katie_pipeline_preview.py before this fix: a
    # coefficient of 0.039 drifted to +260). Restricting Fisher
    # information to just the terms STLSQ/BIC actually kept gives a
    # well-conditioned, physically meaningful uncertainty estimate.
    active_idx = np.where(np.any(Xi != 0, axis=1))[0]
    F_active = compute_fisher_information(Theta[:, active_idx], noise_std)
    Sigma = np.linalg.inv(F_active + 1e-8 * np.eye(F_active.shape[0]))

    if verbose:
        print(f"=== Identification: {persona.name} "
              f"({len(train_idx)} train / {len(val_idx)} val patients) ===")
        print(f"  Optimal lambda: {lambda_opt:.4f}")
        print(f"  Validation NMAE: {nmae:.4f}  RMSE: {rmse:.4f}  "
              f"[{'PASS' if nmae < 1.0 else 'FAIL'}, accept if NMAE < 1.0]")
        print(f"  Fisher information condition number "
              f"(full library, {Theta.shape[1]} terms): {np.linalg.cond(F):.2e}")
        print(f"  Fisher information condition number "
              f"(active terms only, {len(active_idx)}): {np.linalg.cond(F_active):.2e}\n")

    return IdentificationResult(persona=persona, dataset=dataset, Theta=Theta, dX=dX,
                                   lib_labels=lib_labels, protected_idx=protected_idx,
                                   lambda_opt=lambda_opt, Xi=Xi, nmae=nmae, rmse=rmse,
                                   F=F, Sigma=Sigma, active_idx=active_idx)


def select_uncertain_terms(Xi, protected_idx, top_k=2) -> List[Tuple[int, int]]:
    """Top-K (by |identified coefficient|) nonzero pure-linear action
    terms per dynamic state -- see module docstring."""
    xi_idx_pairs = []
    for state in (0, 1):
        candidates = np.where(protected_idx & (Xi[:, state] != 0))[0]
        order = candidates[np.argsort(-np.abs(Xi[candidates, state]))]
        for term_idx in order[:top_k]:
            xi_idx_pairs.append((int(term_idx), state))
    return xi_idx_pairs


def seed_cov_P(Sigma, Theta, dX, xi_idx_pairs, active_idx):
    """De-normalise Sigma's diagonal (Cramer-Rao bound, normalised
    library-column space) into physical-coefficient variance, using the
    same col_std/dX_std convention as stlsq.py/prune_sindy_bic.py.

    Sigma is indexed over `active_idx` (active-terms-only Fisher
    submatrix, see run_identification), not the full library, so
    term_idx (a full-library column index from xi_idx_pairs) must be
    mapped to its position within active_idx first."""
    col_std = Theta.std(axis=0, ddof=1)
    col_std[col_std < 1e-10] = 1.0
    dX_std = dX.std(axis=0, ddof=1)
    dX_std[dX_std < 1e-10] = 1.0

    active_pos = {int(idx): pos for pos, idx in enumerate(active_idx)}

    variances = []
    for term_idx, state_idx in xi_idx_pairs:
        sigma_idx = active_pos[term_idx]
        scale = dX_std[state_idx] / col_std[term_idx]
        variances.append(max(Sigma[sigma_idx, sigma_idx] * scale ** 2, 1e-6))
    return np.diag(variances)


def run_dual_control_planning(ident, xi_idx_pairs, n_sessions=12,
                                 max_du_iterations=60, first_run_max_du_iterations=120,
                                 rng=None, verbose=False, u_bar_0=None):
    if rng is None:
        rng = np.random.default_rng()

    persona = ident.persona
    n_p = len(xi_idx_pairs)
    xi_idx = np.array(xi_idx_pairs, dtype=int) if n_p > 0 else np.zeros((0, 2), dtype=int)

    # a fresh synthetic patient of the SAME persona -- the one being planned for
    test_cfg = {"n_patients": 1, "n_sessions": n_sessions, "noise_std": 0.0,
                 "persona": persona}
    test_ds = generate_synthetic_dataset(test_cfg, rng=rng)
    x0 = test_ds.X[:, 0, 0]
    swat = float(test_ds.swat_ceiling[0])

    p_hat_0 = np.array([ident.Xi[t, s] for t, s in xi_idx_pairs]) if n_p > 0 else np.zeros(0)
    p_true = np.array([test_ds.Xi_true[t, s] for t, s in xi_idx_pairs]) if n_p > 0 else np.zeros(0)
    cov_P = seed_cov_P(ident.Sigma, ident.Theta, ident.dX, xi_idx_pairs, ident.active_idx) if n_p > 0 \
        else np.zeros((0, 0))

    constants = SindyModelConstants(
        Xi=ident.Xi.copy(), lib_cfg=LIB_CFG, swat_ceiling=swat, xi_idx=xi_idx,
        F_dyn=1e-3, F_param=1e-4, persona=persona_weights(persona))

    u_lims = np.tile(np.array([[0.0, 1.0]]), (NU, 1))
    dt = 1.0

    result = main_outer_control_loop(
        T=n_sessions * dt, dt=dt, x_hat_0=x0, x_true_0=x0.copy(),
        p_hat_0=p_hat_0, p_true=p_true, cov_X=1e-3, cov_P=cov_P,
        constants=constants, u_lims=u_lims, u_lim_method=2,
        dynamics=discrete_state_dynamics, measurement=measurement, cost=l_cost,
        continuous_dynamics=continuous_state_dynamics, ny=2, nv=2, nw=NX + n_p,
        reg_type=1, max_du_iterations=max_du_iterations,
        first_run_max_du_iterations=first_run_max_du_iterations,
        augment_states_in_ilqg=True, augment_states_in_filter=True,
        horizon_mode=1, u_bar_0=u_bar_0, verbose=verbose)

    return result, p_true, swat


def print_control_report(ident, result, xi_idx_pairs, p_true, swat):
    x_true, x_hat, p_hat, u = result["x_true"], result["x_hat"], result["p_hat"], result["u"]
    N = x_true.shape[1]

    print(f"=== Dual-control therapy plan: {ident.persona.name} "
          f"(recovery ceiling swat={swat:.3f}) ===")
    print(f"{'Session':>7} | {'SCIM (true)':>11} | {'BBS (true)':>10} | {'mean dose':>9}")
    print("-" * 48)
    for k in range(N):
        dose_str = f"{(0.5 * np.tanh(u[:, k]) + 0.5).mean():9.3f}" if k < N - 1 else " " * 9
        print(f"{k:7d} | {x_true[0, k]:11.3f} | {x_true[1, k]:10.3f} | {dose_str}")

    if xi_idx_pairs:
        print("\nDual-estimated coefficients (identified -> refined vs. true):")
        for i, (term_idx, state_idx) in enumerate(xi_idx_pairs):
            label = ident.lib_labels[term_idx].split(" [s=")[0]
            state_name = ident.dataset.state_names[state_idx]
            print(f"  {label} (d{state_name}/dt): "
                  f"identified={p_hat[i, 0]:+.4f} -> "
                  f"final estimate={p_hat[i, -1]:+.4f}  "
                  f"(true={p_true[i]:+.4f})")
    print(f"\nTotal true cost over horizon: {result['total_true_cost']:.4f}\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ident = run_identification("frail_elder", n_patients=60, n_sessions=12,
                                  noise_std=0.03, rng=rng, verbose=True)

    print_identified_equations(ident.Xi, ident.lib_labels, ident.dataset.state_names, cfg={})

    # top_k=1 (n_p=2 total: one dominant therapy coefficient per dynamic
    # state) -- forward_pass.py's cost-Hessian finite difference is O(n^2)
    # in the augmented dimension nxa+nu, so keeping n_p small matters for
    # runtime; a shorter 8-week control horizon for the same reason
    # (linear in the number of weeks instead).
    xi_idx_pairs = select_uncertain_terms(ident.Xi, ident.protected_idx, top_k=1)

    result, p_true, swat = run_dual_control_planning(
        ident, xi_idx_pairs, n_sessions=8, max_du_iterations=30,
        first_run_max_du_iterations=60, rng=rng, verbose=False)

    print_control_report(ident, result, xi_idx_pairs, p_true, swat)
