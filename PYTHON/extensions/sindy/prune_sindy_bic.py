"""
Post-selection BIC pruning of a SINDy model, per dynamic state equation.

Ported from Katie's CODES/KATIE'S CODE/.../prune_sindy_bic.m.

After stlsq.py converges (using cross_validate_lambda.py's single
GLOBAL lambda, applied uniformly across every state equation), this
runs a SECOND, per-equation refinement pass: for each of the two
dynamic states (SCIM, BBS -- states 3-6 are fixed patient attributes
whose dynamics are forced to zero elsewhere, so pruning them is
meaningless), it greedily removes the smallest-magnitude currently-
active UNPROTECTED term one at a time, refitting after each removal,
and accepts the removal only if BIC does not meaningfully increase:

    BIC = N * log(RSS/N) + k * log(N)

BIC formalises Occam's razor: rewards lower residual (better fit),
penalises term count k by log(N). This catches a failure mode a single
global STLSQ lambda can't: a term whose raw coefficient magnitude
happened to clear the CV-selected lambda (so STLSQ kept it) but that
barely improves the fit relative to the complexity cost it adds. The
0.5-unit acceptance cushion (bic_trial <= bic_curr + 0.5) exists so
pruning doesn't get stuck rejecting removals over floating-point-scale
BIC noise -- it only refuses removals that meaningfully hurt BIC.

INPUTS
  Xi            (n_terms, n_states)  sparse coefficients (physical units,
                as produced by stlsq.py)
  Theta         (N, n_terms)         library matrix (physical units)
  dX            (N, n_states)        state derivatives
  lib_labels    list[str] of length n_terms  library term labels (with
                "[s=...]" scale tags, as produced by build_sindy_library.py)
  state_names   list[str] of length n_states  state variable names
  verbose       bool
  protected_idx (n_terms,) bool or None  terms immune to pruning
                (None -> all False)

OUTPUTS
  Xi_pruned    (n_terms, n_states)  pruned coefficient matrix
  bic_history  list[dict]  one entry per PRUNED state (states beyond the
               first two are skipped entirely, so this can have fewer
               than n_states entries), each with keys 'state' (str),
               'bic_vals' (np.ndarray), 'nnz_vals' (np.ndarray)

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - Same ddof=1 (N-1 std) correctness trap as build_sindy_library.py/
    stlsq.py/cross_validate_lambda.py: `std(Theta,0,1)`/`std(dX,0,1)`
    need ddof=1, not NumPy's default ddof=0.
  - `min(2, n_states)` (dynamic-states-only) is a hardcoded SCI-domain
    assumption baked directly into the MATLAB source, not a bug --
    matches main_sindy_pipeline.m's separate "zero static state
    equations" step. Kept as-is; flagged since it would need updating
    if a future model ever added more genuinely dynamic states.
  - bic_history is a list of plain dicts here instead of a MATLAB
    struct array -- same information, more idiomatic Python. Nothing
    in the currently-ported pipeline consumes it yet (would feed
    plot_results.m, not yet ported).
"""

import re

import numpy as np


def prune_sindy_bic(Xi, Theta, dX, lib_labels, state_names, verbose,
                       protected_idx=None):
    N, n_terms = Theta.shape
    n_states = dX.shape[1]

    if protected_idx is None:
        protected_idx = np.zeros(n_terms, dtype=bool)
    protected_idx = np.asarray(protected_idx, dtype=bool).ravel()

    Xi_pruned = Xi.copy()
    bic_history = []

    # ---- internal normalisation (matches stlsq.py's convention) ----
    col_std = Theta.std(axis=0, ddof=1)
    col_std[col_std < 1e-10] = 1.0
    Theta_n = Theta / col_std

    dX_std = dX.std(axis=0, ddof=1)
    dX_std[dX_std < 1e-10] = 1.0
    dX_n = dX / dX_std

    eps = np.finfo(float).eps

    # ---- BIC pruning per dynamic state (SCIM and BBS only) ----
    for s in range(min(2, n_states)):

        active = np.where(Xi_pruned[:, s] != 0)[0]
        if active.size == 0:
            continue

        # convert the already-fitted PHYSICAL coefficients back to
        # normalised space (inverse of stlsq.py's own de-normalisation)
        scale_s = dX_std[s] / col_std  # (n_terms,)
        xi_s_n = Xi_pruned[:, s] / scale_s

        bic_trace = [None]
        nnz_trace = [None]

        dX_pred_n = Theta_n[:, active] @ xi_s_n[active]
        rss = np.sum((dX_n[:, s] - dX_pred_n) ** 2)
        bic_curr = N * np.log(rss / N + eps) + active.size * np.log(N)

        bic_trace[0] = bic_curr
        nnz_trace[0] = active.size

        if verbose:
            print(f"\n  BIC pruning: d({state_names[s]})/dt")
            print(f"    Initial: {active.size} terms, BIC = {bic_curr:.2f}")

        # ---- greedy backward elimination -- never drop protected terms ----
        keep_pruning = True
        while keep_pruning and active.size > 1:

            droppable = active[~protected_idx[active]]
            if droppable.size == 0:
                if verbose:
                    print("    Stopped (only protected terms remain)")
                break

            drop_local = np.argmin(np.abs(xi_s_n[droppable]))
            candidate_drop = droppable[drop_local]
            trial_active = active[active != candidate_drop]

            Theta_tr = Theta_n[:, trial_active]
            ridge_t = 1e-8 * np.eye(trial_active.size)
            xi_trial_n = np.zeros(n_terms)
            xi_trial_n[trial_active] = np.linalg.solve(
                Theta_tr.T @ Theta_tr + ridge_t, Theta_tr.T @ dX_n[:, s])

            dX_pred_trial = Theta_n[:, trial_active] @ xi_trial_n[trial_active]
            rss_trial = np.sum((dX_n[:, s] - dX_pred_trial) ** 2)
            bic_trial = N * np.log(rss_trial / N + eps) + trial_active.size * np.log(N)

            raw_lbl = _strip_scale_tag(lib_labels[candidate_drop])

            if bic_trial <= bic_curr + 0.5:  # accept: BIC does not meaningfully increase
                active = trial_active
                xi_s_n = xi_trial_n
                bic_curr = bic_trial

                bic_trace.append(bic_curr)
                nnz_trace.append(active.size)

                if verbose:
                    print(f'    Dropped "{raw_lbl}" -> {active.size} terms, '
                          f"BIC = {bic_trial:.2f}")
            else:
                keep_pruning = False
                if verbose:
                    print(f'    Stopped (dropping "{raw_lbl}" would increase '
                          f"BIC by {bic_trial - bic_curr:.2f})")

        # ---- write pruned coefficients back to physical Xi ----
        xi_phys = np.zeros(n_terms)
        xi_phys[active] = xi_s_n[active] * scale_s[active]
        Xi_pruned[:, s] = xi_phys

        bic_history.append(dict(state=state_names[s],
                                  bic_vals=np.array(bic_trace),
                                  nnz_vals=np.array(nnz_trace)))

        if verbose:
            print(f"    Final: {active.size} terms retained (from "
                  f"{np.count_nonzero(Xi[:, s])})")

    return Xi_pruned, bic_history


def _strip_scale_tag(label):
    return re.sub(r"\s*\[s=[\d.]+\]", "", label)
