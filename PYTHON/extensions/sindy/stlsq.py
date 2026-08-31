"""
Sequential Thresholded Least Squares (STLSQ) for SINDy sparse regression.

Ported from Katie's CODES/KATIE'S CODE/.../stlsq.m.

Recovers a sparse coefficient matrix Xi such that dX ~= Theta @ Xi, by
alternating ordinary least squares with hard-thresholding of small
(normalised) coefficients, until the nonzero pattern stabilises:

  1. Normalise every column of Theta and dX by its own std -- makes a
     single threshold `lam` meaningful across columns of very different
     physical scale.
  2. Initial ridge-regularised least-squares fit on the FULL library.
  3. Iterate: threshold (zero any |Xi| < lam, EXCEPT protected_idx
     terms, which are never zeroed regardless of magnitude) -> per-state
     refit on the surviving active columns only -> if thresholding drops
     a state below min_terms active terms, revive the highest-magnitude
     previously-dropped unprotected terms until the floor is met -> stop
     once the relative Frobenius-norm change falls below tol.
  4. Hard cap: if max_terms is finite, greedily remove the smallest-
     magnitude UNPROTECTED active term one at a time (refitting after
     each removal) until every state equation has <= max_terms terms.
  5. De-normalise back to physical units.

protected_idx / min_terms / max_terms are exactly the customisations
that motivated hand-porting this instead of using pysindy's STLSQ
optimizer (see CLAUDE.md's SINDy library decision) -- none of the three
exist in pysindy or sklearn.

INPUTS
  Theta         (N, n_terms)   library matrix (physical units)
  dX            (N, n_states)  state derivatives
  lam           scalar         sparsity threshold (normalised space)
  max_iter      int            maximum iterations
  tol           float          convergence tolerance
  verbose       bool           print iteration info
  max_terms     float or None  hard cap on active terms (None -> inf)
  protected_idx (n_terms,) bool or None  terms immune to thresholding
                (None -> all False)
  min_terms     int            minimum active terms per state (default 1)

OUTPUT
  Xi  (n_terms, n_states)  sparse coefficient matrix (physical units)

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - MATLAB's `lambda` is named `lam` here (Python keyword), same
    convention already used throughout this port (ilqg_function.py, etc.)
  - MATLAB's `std(Theta,0,1)` / `std(dX,0,1)` default to N-1 (unbiased)
    normalisation; this port uses ddof=1 explicitly to match -- the same
    correctness trap already flagged in build_sindy_library.py. Getting
    this wrong would silently rescale every threshold decision.
"""

import numpy as np


def stlsq(Theta, dX, lam, max_iter, tol, verbose, max_terms=None,
           protected_idx=None, min_terms=1):
    n_terms = Theta.shape[1]
    n_states = dX.shape[1]

    if max_terms is None:
        max_terms = np.inf
    if protected_idx is None:
        protected_idx = np.zeros(n_terms, dtype=bool)
    protected_idx = np.asarray(protected_idx, dtype=bool).ravel()

    # ---- internal normalisation ----
    col_std = Theta.std(axis=0, ddof=1)
    col_std[col_std < 1e-10] = 1.0
    Theta_n = Theta / col_std

    dX_std = dX.std(axis=0, ddof=1)
    dX_std[dX_std < 1e-10] = 1.0
    dX_n = dX / dX_std

    # ---- initial ridge-regularised least squares ----
    ridge = 1e-6 * np.eye(n_terms)
    Xi_n = np.linalg.solve(Theta_n.T @ Theta_n + ridge, Theta_n.T @ dX_n)

    if verbose:
        nnz_init = np.sum(np.abs(Xi_n) >= lam)
        print(f"  STLSQ: initial nnz = {nnz_init} / {Xi_n.size}  (lambda = {lam:.4f})")

    # ---- STLSQ iterations ----
    for it in range(1, max_iter + 1):
        Xi_prev = Xi_n.copy()

        small = (np.abs(Xi_n) < lam) & ~protected_idx[:, None]
        Xi_n[small] = 0.0

        for s in range(n_states):
            active = np.where(~small[:, s] | (Xi_n[:, s] != 0))[0]

            if np.sum(Xi_n[:, s] != 0) < min_terms:
                inactive = np.where((Xi_n[:, s] == 0) & ~protected_idx)[0]
                order = np.argsort(-np.abs(Xi_prev[inactive, s]))
                needed = min_terms - np.sum(Xi_n[:, s] != 0)
                revive = inactive[order[:min(needed, len(order))]]
                Xi_n[revive, s] = Xi_prev[revive, s]
                active = np.where(Xi_n[:, s] != 0)[0]

            if active.size == 0:
                Xi_n[:, s] = 0.0
                continue

            Theta_active = Theta_n[:, active]
            ridge_act = 1e-8 * np.eye(active.size)
            xi_act = np.linalg.solve(Theta_active.T @ Theta_active + ridge_act,
                                       Theta_active.T @ dX_n[:, s])
            Xi_n[:, s] = 0.0
            Xi_n[active, s] = xi_act

        delta = (np.linalg.norm(Xi_n - Xi_prev, "fro")
                  / (np.linalg.norm(Xi_prev, "fro") + np.finfo(float).eps))
        if verbose and it % 10 == 0:
            print(f"  Iter {it:3d} | delta = {delta:.2e} | nnz = {np.count_nonzero(Xi_n)}")
        if delta < tol:
            if verbose:
                print(f"  Converged at iter {it} (delta = {delta:.2e})")
            break

    # ---- hard cap: prune to max_terms per state equation ----
    if np.isfinite(max_terms):
        for s in range(n_states):
            active = np.where(Xi_n[:, s] != 0)[0]
            while active.size > max_terms:
                active_unprotected = active[~protected_idx[active]]
                if active_unprotected.size == 0:
                    break

                prune_local = np.argmin(np.abs(Xi_n[active_unprotected, s]))
                prune_global = active_unprotected[prune_local]
                active = active[active != prune_global]

                if active.size == 0:
                    Xi_n[:, s] = 0.0
                    break
                Theta_active = Theta_n[:, active]
                ridge_act = 1e-8 * np.eye(active.size)
                xi_act = np.linalg.solve(Theta_active.T @ Theta_active + ridge_act,
                                           Theta_active.T @ dX_n[:, s])
                Xi_n[:, s] = 0.0
                Xi_n[active, s] = xi_act
        if verbose:
            print(f"  After max_terms cap: nnz = {np.count_nonzero(Xi_n)} / {Xi_n.size}")

    # ---- de-normalise ----
    Xi = Xi_n * (dX_std[None, :] / col_std[:, None])

    if verbose:
        print(f"  STLSQ: final nnz = {np.count_nonzero(Xi)} / {Xi.size}")

    return Xi
