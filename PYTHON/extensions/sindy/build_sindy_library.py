"""
Constructs the SINDy candidate function library Theta and the
corresponding state-derivative matrix dX from trajectory data.

Ported from Katie's CODES/KATIE'S CODE/.../build_sindy_library.m.

This is step 1+2 of SINDy identification (dX ~= Theta @ Xi): build the
regression PROBLEM (features + targets), not the sparse solve itself
(that's stlsq.py, not yet ported).

  1. Numerical differentiation: estimate ẋ at every session via finite
     differences -- central difference at interior sessions (2nd-order
     accurate), forward/backward difference only at the two trajectory
     endpoints (1st-order, noisier).
  2. Observation stacking: keep ONLY interior sessions (drop both
     endpoints) when flattening (patient, session) pairs into rows --
     specifically because the endpoint derivatives are the less-
     accurate one-sided estimates.
  3. Candidate library: bias, linear terms in [x; u], squared terms,
     all pairwise cross-products (if cfg['include_cross']), cubic terms
     for just the first two state variables (SCIM, BBS -- kept narrow
     so the library doesn't combinatorially explode), optional sin/cos.
     Each column is tagged with its standard deviation (for later
     de-normalised equation reporting) and protected_idx flags columns
     that must survive STLSQ thresholding regardless of fitted
     coefficient size: by default the raw action terms a1..am, plus
     anything named in cfg['protected_terms'].

INPUTS
  X   (n_states, n_sessions, n_patients)  state trajectories
  A   (n_actions, n_sessions, n_patients) actions
  cfg  dict with keys 'poly_order' (int), 'include_cross' (bool),
       'include_trig' (bool, REQUIRED -- see deviation note below),
       optionally 'protected_terms' (list[str]), 'verbose' (bool)

OUTPUTS
  Theta          (N, n_terms)  library matrix (N = total observations)
  dX             (N, n_states) state derivatives (finite differences)
  labels         list[str] of length n_terms, each tagged "name [s=std]"
  protected_idx  (n_terms,) bool array

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - cfg['include_trig'] is accessed WITHOUT a default (raises KeyError
    if missing), matching the MATLAB source's own unconditional
    `if cfg.include_trig` (no isfield guard) -- unlike
    ContinuousStateDynamics.m's OWN copy of a similar row-builder,
    which defensively checks `isfield(cfg,'include_trig')`. This
    asymmetry exists in the MATLAB source itself; flagged, not
    silently smoothed over.
  - The docstring in the MATLAB source claims specific "clinically
    motivated" cross-terms (x1*x3, x2*x3, a1*x3, x6*a1) are
    "prioritised" -- but the code just enumerates EVERY pairwise
    cross-product uniformly; nothing treats those four specially. The
    actual survival guarantee comes later, via cfg['protected_terms']
    -- and the terms actually configured there in main_sindy_pipeline.m
    ('x3*a3', 'x6*a8') don't even match the docstring's illustrative
    list. Cosmetic/doc drift in the MATLAB source, not a functional bug.
  - The finite-difference loop is vectorised across patients (no
    semantic change -- MATLAB looped patient-by-patient only because
    the endpoint slices differ from the interior slices, not because
    patients need separate handling).
  - MATLAB's `std(Theta, 0, 1)` defaults to N-1 (unbiased/sample)
    normalisation -- NumPy's std() defaults to N (population,
    ddof=0). This port uses ddof=1 explicitly to match; getting this
    wrong would silently scale every column tag (and, later, every
    stlsq.py threshold) by sqrt(N/(N-1)).
"""

import re

import numpy as np


def build_sindy_library(X, A, cfg):
    n_states, n_sessions, n_patients = X.shape
    n_actions = A.shape[0]
    dt = 1.0  # 1-week session interval

    # ---- finite differences ----
    dX_full = np.zeros((n_states, n_sessions, n_patients))
    dX_full[:, 0, :] = (X[:, 1, :] - X[:, 0, :]) / dt
    for s in range(1, n_sessions - 1):
        dX_full[:, s, :] = (X[:, s + 1, :] - X[:, s - 1, :]) / (2 * dt)
    dX_full[:, -1, :] = (X[:, -1, :] - X[:, -2, :]) / dt

    # ---- stack observations (interior sessions only) ----
    obs_sessions = range(1, n_sessions - 1)
    N = n_patients * len(list(obs_sessions))

    Z = np.zeros((N, n_states + n_actions))
    dX = np.zeros((N, n_states))

    row = 0
    for p in range(n_patients):
        for s in obs_sessions:
            Z[row, :] = np.concatenate([X[:, s, p], A[:, s, p]])
            dX[row, :] = dX_full[:, s, p]
            row += 1

    # ---- build library columns ----
    var_names = [f"x{i + 1}" for i in range(n_states)] + \
                [f"a{j + 1}" for j in range(n_actions)]
    n_vars = n_states + n_actions

    Theta_cols = [np.ones(N)]
    labels = ["1"]

    for i in range(n_vars):
        Theta_cols.append(Z[:, i])
        labels.append(var_names[i])

    poly_order = cfg["poly_order"]
    if poly_order >= 2:
        for i in range(n_vars):
            Theta_cols.append(Z[:, i] ** 2)
            labels.append(f"{var_names[i]}^2")
        if cfg["include_cross"]:
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    Theta_cols.append(Z[:, i] * Z[:, j])
                    labels.append(f"{var_names[i]}*{var_names[j]}")

    if poly_order >= 3:
        for i in range(min(n_states, 2)):  # x1^3, x2^3 only
            Theta_cols.append(Z[:, i] ** 3)
            labels.append(f"{var_names[i]}^3")

    if cfg["include_trig"]:
        for i in range(n_vars):
            Theta_cols.append(np.sin(Z[:, i]))
            labels.append(f"sin({var_names[i]})")
        for i in range(n_vars):
            Theta_cols.append(np.cos(Z[:, i]))
            labels.append(f"cos({var_names[i]})")

    Theta = np.column_stack(Theta_cols)  # (N, n_terms)

    # column std (N-1 normalisation, matches MATLAB's std(Theta,0,1))
    col_std = Theta.std(axis=0, ddof=1)
    col_std[col_std < 1e-10] = 1.0  # constant (bias) column gets scale = 1

    labels = [f"{lbl} [s={col_std[k]:.3f}]" for k, lbl in enumerate(labels)]

    # ---- protected_idx ----
    n_terms = len(labels)
    raw_labels_clean = [re.sub(r"\s*\[s=[\d.]+\]", "", lbl) for lbl in labels]

    default_protect_exact = [f"a{j + 1}" for j in range(n_actions)]
    user_protect_exact = list(cfg.get("protected_terms", []))
    all_protect_exact = set(default_protect_exact + user_protect_exact)

    protected_idx = np.array([lbl in all_protect_exact for lbl in raw_labels_clean])

    if cfg.get("verbose", False):
        print(f"  Protected terms: {protected_idx.sum()} "
              "(action terms + clinician-mandated interactions)")

    return Theta, dX, labels, protected_idx
