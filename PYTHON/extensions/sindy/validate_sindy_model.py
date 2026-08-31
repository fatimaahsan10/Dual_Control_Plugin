"""
Validate the identified SINDy model by open-loop forward simulation on
held-out patient trajectories, scored via NMAE/RMSE.

Ported from Katie's CODES/KATIE'S CODE/.../validate_sindy_model.m.

build_sindy_library -> stlsq -> cross_validate_lambda -> prune_sindy_bic
together select a sparse model that fits the TRAINING data one
derivative-step at a time. None of that checks whether rolling the
model forward from a real patient's admission state, driven by their
real prescribed therapy doses, actually tracks their real recovery
curve. This file does: it forward-simulates OPEN-LOOP / AUTOREGRESSIVE
(the model's own predicted previous state, not the true observed state,
feeds the next step) via Euler integration, x += Theta(x,u) @ Xi,
clipped to [0,1]. Small per-step errors compound over the horizon here
in a way one-step-ahead residual fitting can't expose -- a model that
looked fine under STLSQ/CV/BIC can still be unstable or badly biased
once simulated forward. This is the manuscript's accept/reject gate
(Section IV-B: accept if NMAE < 1.0).

INPUTS
  Xi          (n_terms, n_states)  sparse coefficient matrix (physical units)
  X_val       (n_states, n_sessions, n_val)   validation states
  A_val       (n_actions, n_sessions, n_val)  validation actions
  Y_val       unused (see deviation note)
  lib_labels  unused (see deviation note)
  cfg         dict with 'poly_order', 'include_cross', 'include_trig'
              (matches build_sindy_library.py's cfg)

OUTPUTS
  nmae             scalar  mean absolute error / dynamic range of X_val
  rmse             scalar  root mean squared error
  sim_trajectories dict with keys 'X_true', 'X_sim', 'n_val',
                   'n_sessions', 'state_idx' -- for downstream plotting
                   (plot_results.m, not yet ported)

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - Y_val and lib_labels are genuinely unused in the MATLAB source's own
    function body -- kept as parameters for call-site signature fidelity
    with main_sindy_pipeline.m, same convention as Measurement.m's
    unused `u`/`c` and l_cost.m's unused `tracking_trajectory`.
  - Reuses continuous_state_dynamics.py's build_sindy_row (now public)
    instead of re-implementing a third independent copy of the
    row-builder. This DEVIATES from the MATLAB source's own local
    build_library_row, whose cubic-term count uses
    `n_states_approx = length(z) - 3` -- a leftover assumption that
    n_actions is always 3, from the older 3-control demo. At Katie's
    real dims (6 states, 13 actions -> length(z)=19, n_states_approx=16)
    this happens to still give the correct cubic-term count purely
    because `min(n_states_approx, 2)` clamps regardless of how wrong
    n_states_approx is -- but it's a latent bug for any other
    state/action count, since it would silently misalign which library
    columns Xi's coefficients are being multiplied against.
    ContinuousStateDynamics.m's OWN copy of this row-builder already
    uses the correct `length(x)`; this port uses that (shared, tested)
    implementation instead of replicating the fragile approximation a
    second time.
  - No `swat` recovery-ceiling clamp here (unlike
    continuous_state_dynamics.py) -- just a plain [0,1] box clip. This
    is intentional, not an inconsistency: this file evaluates the RAW
    identified model's open-loop predictive accuracy against real data,
    independent of the controller's own dynamics clamping. Similarly,
    static states (3-6) are expected to stay constant only because the
    caller (main_sindy_pipeline.m's "zero static state equations" step)
    already zeroed their Xi columns -- this file doesn't independently
    enforce that itself.
"""

import numpy as np

# Layering note: this reaches "up" into applications/sci_rehab for
# build_sindy_row rather than duplicating it a third time (see
# continuous_state_dynamics.py's own docstring on the two other
# independent copies of this row-builder). A generic extensions/sindy
# module depending on a specific application is a real inversion of the
# core -> extensions -> applications layering; flagged here rather than
# silently normalized, per this project's existing discrepancy convention.
from applications.sci_rehab.continuous_state_dynamics import build_sindy_row


def validate_sindy_model(Xi, X_val, A_val, Y_val, lib_labels, cfg):
    n_states, n_sessions, n_val = X_val.shape

    X_sim = np.zeros((n_states, n_sessions, n_val))

    for p in range(n_val):
        x_sim = X_val[:, 0, p].copy()
        X_sim[:, 0, p] = x_sim

        for s in range(n_sessions - 1):
            a_s = A_val[:, s, p]

            theta_row = build_sindy_row(x_sim, a_s, cfg)
            dx_pred = theta_row @ Xi  # (n_states,)

            x_sim = x_sim + dx_pred
            x_sim = np.clip(x_sim, 0.0, 1.0)

            X_sim[:, s + 1, p] = x_sim

    # ---- error metrics: focus on the two dynamic states (SCIM, BBS) ----
    dynamic_idx = [0, 1]
    X_true_dyn = X_val[dynamic_idx, :, :]
    X_sim_dyn = X_sim[dynamic_idx, :, :]
    err = X_true_dyn - X_sim_dyn

    rmse = np.sqrt(np.mean(err ** 2))

    dyn_range = X_true_dyn.max() - X_true_dyn.min()
    if dyn_range < np.finfo(float).eps:
        dyn_range = 1.0
    nmae = np.mean(np.abs(err)) / dyn_range

    sim_trajectories = dict(X_true=X_val, X_sim=X_sim, n_val=n_val,
                              n_sessions=n_sessions, state_idx=dynamic_idx)

    return nmae, rmse, sim_trajectories
