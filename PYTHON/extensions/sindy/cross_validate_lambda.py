"""
Select the STLSQ sparsity threshold `lambda` via 5-fold cross-validation.

Ported from Katie's CODES/KATIE'S CODE/.../cross_validate_lambda.m.

Lambda trades off model fit against sparsity: too small and noise-fit
terms survive; too large and it zeroes real dynamics -- including,
historically, every action term. This wraps stlsq.py in a CV loop over
a lambda grid and picks the value that generalises best to held-out
data, biased toward sparser models on near-ties.

  1. Lambda grid: log-spaced 0.001 -> 0.35. The upper bound is
     deliberately held below 0.5 -- letting the grid reach 0.5
     previously let CV selection drift to the ceiling and strip every
     action term.
  2. Data-adaptive floor: a quick full-data ridge OLS fit gives
     "typical" coefficient magnitudes; the grid is trimmed to
     lambda >= max(10% of median |nonzero OLS coeff|, 0.005), so the
     low end of the grid isn't so permissive that sparsity becomes
     uninformative for CV to discriminate on.
  3. K-fold loop: for each lambda, for each of 5 folds, fit stlsq on
     the training fold, score on the held-out fold via NMAE (mean
     absolute residual normalised by the fold's own dynamic range, per
     state, averaged). A fold where every UNPROTECTED term got zeroed
     (degenerate -- only the forced-protected terms survive) gets a
     fixed large penalty instead of a real score.
  4. Complexity penalty: mean CV error is nudged up by
     3%(mean valid error) * (mean_nnz/max_nnz) -- small enough to only
     break near-ties, not override real fit-quality differences. This
     is what actually implements the sparsity preference.
  5. Selection: a deliberately INVERTED 1-SE rule. Find the lambda with
     minimum penalised error, then among all lambdas within 1 SE of
     that minimum, pick the SMALLEST lambda -- not the largest/sparsest,
     which is what a textbook 1-SE rule picks. That direction was
     explicitly abandoned (same ceiling-drift failure as point 1);
     sparsity preference is left entirely to the complexity penalty.
  6. Safety net: refit at lambda_opt on the FULL dataset; if that's
     still degenerate, fall back to the minimum-penalised-error lambda
     regardless of the 1-SE band.

INPUTS
  Theta         (N, n_terms)   library matrix (physical units)
  dX            (N, n_states)  derivative matrix
  cfg           dict with keys 'n_lambda', 'max_iter', 'tol', optionally
                'max_terms', 'min_terms', 'verbose'
  protected_idx (n_terms,) bool or None  (optional) protected term flags
  random_state  int or None  fold-assignment seed (NOT in the MATLAB
                signature -- see deviation note)

OUTPUT
  lambda_opt  scalar  optimal sparsity threshold

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - crossvalind('Kfold', N, n_folds) (MATLAB Statistics Toolbox) is
    replaced with sklearn.model_selection.KFold, per the existing
    "MATLAB-Stats-Toolbox shims" decision in CLAUDE.md's Katie
    conversion plan. MATLAB's fold assignment draws from implicit
    global RNG state with no exposed seed; this port adds an explicit
    `random_state` parameter (not present in the MATLAB signature)
    since deterministic, testable fold assignment has no MATLAB
    equivalent to preserve -- exact fold-membership parity with MATLAB
    was never available to begin with, only "is this a valid k-fold
    CV", which sklearn's KFold provides.
  - The diagnostic CV-curve plot (figure/yyaxis/xline/...) is UI-only
    and doesn't affect the returned value -- dropped, same as every
    other graphics() call already dropped elsewhere in this port
    (ilqg.py, Todorov_estimator.py, etc.).
  - Same ddof=1 (N-1 std) correctness trap as build_sindy_library.py/
    stlsq.py: `std(cv_error, 0, 2)` also needs ddof=1, not NumPy's
    default ddof=0.
  - MATLAB's `max(lambda_floor, 0.005)` IGNORES NaN (MATLAB's max
    returns the non-NaN argument when one side is NaN -- e.g.
    max(NaN, 0.005) = 0.005), which happens when every OLS coefficient
    is exactly 0 (median of an empty "nonzero" set = NaN in both
    MATLAB and NumPy). NumPy's `max()`/`np.maximum` PROPAGATE NaN
    instead (np.maximum(nan, 0.005) = nan), which would silently
    corrupt the grid-trimming comparison below it. Used `np.fmax`
    (NaN-ignoring, matches MATLAB's max semantics here) instead.
"""

import numpy as np
from sklearn.model_selection import KFold

from extensions.sindy.stlsq import stlsq


def cross_validate_lambda(Theta, dX, cfg, protected_idx=None, random_state=0):
    n_terms = Theta.shape[1]
    n_states = dX.shape[1]
    N = Theta.shape[0]

    if protected_idx is None:
        protected_idx = np.zeros(n_terms, dtype=bool)
    protected_idx = np.asarray(protected_idx, dtype=bool).ravel()

    n_folds = 5
    n_lambda = cfg["n_lambda"]
    DEGENERATE_PENALTY = 10.0

    lambda_raw = np.logspace(-3, np.log10(0.35), n_lambda)

    # ---- data-adaptive lower bound ----
    col_std_cv = Theta.std(axis=0, ddof=1)
    col_std_cv[col_std_cv < 1e-10] = 1.0
    Theta_n_cv = Theta / col_std_cv

    dX_std_cv = dX.std(axis=0, ddof=1)
    dX_std_cv[dX_std_cv < 1e-10] = 1.0
    dX_n_cv = dX / dX_std_cv

    ridge_cv = 1e-6 * np.eye(n_terms)
    xi_ols = np.linalg.solve(Theta_n_cv.T @ Theta_n_cv + ridge_cv,
                               Theta_n_cv.T @ dX_n_cv)

    nonzero_ols = xi_ols[xi_ols != 0]
    lambda_floor = 0.10 * np.median(np.abs(nonzero_ols)) if nonzero_ols.size else np.nan
    lambda_floor = np.fmax(lambda_floor, 0.005)  # NaN-ignoring max, see docstring

    lambda_grid = lambda_raw[lambda_raw >= lambda_floor]
    if lambda_grid.size == 0:
        lambda_grid = lambda_raw

    max_terms = cfg.get("max_terms", np.inf)
    min_terms = cfg.get("min_terms", 1)

    n_grid = lambda_grid.size
    cv_error = np.zeros((n_grid, n_folds))
    cv_nnz = np.zeros((n_grid, n_folds))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_of = np.zeros(N, dtype=int)
    for fold_id, (_, test_idx) in enumerate(kf.split(np.arange(N))):
        fold_of[test_idx] = fold_id

    for li in range(n_grid):
        lam = lambda_grid[li]

        for k in range(n_folds):
            test_mask = (fold_of == k)
            train_mask = ~test_mask

            Theta_tr, dX_tr = Theta[train_mask, :], dX[train_mask, :]
            Theta_te, dX_te = Theta[test_mask, :], dX[test_mask, :]

            Xi_k = stlsq(Theta_tr, dX_tr, lam, cfg["max_iter"], cfg["tol"], False,
                          max_terms, protected_idx, min_terms)

            unprotected_nnz = np.count_nonzero(Xi_k[~protected_idx, :])
            if unprotected_nnz == 0:
                cv_error[li, k] = DEGENERATE_PENALTY
                cv_nnz[li, k] = np.count_nonzero(Xi_k)
                continue

            dX_pred = Theta_te @ Xi_k

            nmae_per_state = np.zeros(n_states)
            for s in range(n_states):
                r_s = dX_te[:, s] - dX_pred[:, s]
                rng_s = dX_te[:, s].max() - dX_te[:, s].min()
                if rng_s < 1e-10:
                    rng_s = 1.0
                nmae_per_state[s] = np.mean(np.abs(r_s)) / rng_s
            cv_error[li, k] = nmae_per_state.mean()
            cv_nnz[li, k] = np.count_nonzero(Xi_k)

    mean_cv = cv_error.mean(axis=1)
    std_cv = cv_error.std(axis=1, ddof=1) / np.sqrt(n_folds)

    # ---- complexity penalty ----
    mean_nnz = cv_nnz.mean(axis=1)
    max_nnz = mean_nnz.max() + np.finfo(float).eps

    valid_mean_cv = mean_cv[mean_cv < DEGENERATE_PENALTY]
    if valid_mean_cv.size == 0:
        raise RuntimeError(
            "All lambda values produced degenerate models. Check that "
            "protected_terms are correct and the data has sufficient variation.")

    alpha_penalty = 0.03 * valid_mean_cv.mean()
    penalised_cv = mean_cv + alpha_penalty * (mean_nnz / max_nnz)

    valid_mask = mean_cv < DEGENERATE_PENALTY
    penalised_valid = penalised_cv.copy()
    penalised_valid[~valid_mask] = np.inf

    min_idx_p = int(np.argmin(penalised_valid))
    min_err_p = penalised_valid[min_idx_p]
    threshold_1se = min_err_p + std_cv[min_idx_p]

    # inverted 1-SE rule: smallest lambda within 1 SE of the minimum
    # (see module docstring for why this isn't the textbook direction)
    candidates = np.where(penalised_valid <= threshold_1se)[0]
    lambda_opt = lambda_grid[candidates[0]]

    # ---- degenerate safety net ----
    Xi_test = stlsq(Theta, dX, lambda_opt, cfg["max_iter"], cfg["tol"], False,
                      max_terms, protected_idx, min_terms)
    if np.count_nonzero(Xi_test[~protected_idx, :]) == 0:
        lambda_opt = lambda_grid[min_idx_p]
        if cfg.get("verbose", False):
            print(f"  [CV] Selected lambda produced degenerate model; "
                  f"using min-error lambda = {lambda_opt:.4f}")

    return lambda_opt
