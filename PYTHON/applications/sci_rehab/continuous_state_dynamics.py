"""
SCI rehabilitation plant dynamics: ẋ = f(x, u), identified by SINDy
rather than hand-specified. Replaces Andrew's SIDARTHE COVID plant.

Ported from Katie's CODES/KATIE'S CODE/.../ContinuousStateDynamics.m.

STATE VECTOR (nx = 6, all normalised to [0, 1]):
    x1: SCIM_norm (independence/function)   x2: BBS_norm (balance)
    x3: AIS grade (fixed)   x4: Age (fixed)   x5: DPI (fixed)
    x6: Caregiver support (fixed)
Only x1/x2 actually evolve; x3-x6 are patient attributes carried in the
state vector for the controller to condition on, not integrated.

ACTION VECTOR: nu WHO therapy doses (see who_therapies.m), each in
[0, 1]. NOTE: this file's MATLAB docstring header says "nu = 3"
(Intensity/Modality/Frequency) -- that's stale, left over from the
older single-persona demo (Outer_Control_Loop.m). The code itself is
action-dimension-generic (nu is read off u's shape / the SINDy library
column count); the current pipeline (run_sgba_personas.m, l_cost.m,
define_persona.m's `assert(numel(therapy_efficacy) == 13)`) uses 13.
Flagged rather than silently corrected in the port.

AUGMENTED PARAMETERS (augment_states=True): xa[nx:] holds uncertain
SINDy coefficients being estimated online by dual iLQG -- curiosity
drives actions that reduce uncertainty (Fp) about these, not just
actions that reduce state cost.

Biological guardrails layered on top of the raw SINDy fit (not implied
by the regression itself):
  - static states (x3-x6) always have zero dynamics, regardless of what
    the identified model says about them
  - SCIM/BBS growth is clamped at a floor of 0 and a patient-specific
    recovery ceiling `swat` -- a fitted polynomial has no reason to
    respect either boundary on its own.

DEVIATION FROM THE LITERAL MATLAB SOURCE: the MATLAB reads its SINDy
model (Xi, lib_cfg, swat_ceiling, xi_idx, F_dyn, F_param) from a
`global SINDY_MODEL` and never uses its own `c` argument. This port
reads the model from the explicit `constants` argument instead -- see
sindy_model_constants.py's module docstring for why.
"""

import numpy as np


def continuous_state_dynamics(dt, xa, u, constants, augment_states, u_lims,
                                 u_lim_method):
    """
    Returns
    -------
    xdot : (nx [+ n_p], N+1)   state (and, if augmented, parameter) derivatives
    nx   : 6
    n_p  : number of augmented (estimated) SINDy coefficients, 0 if not augmented
    F    : (nx, nx)   dynamics noise scale matrix (F_dyn * I)
    Fp   : (n_p, n_p) parameter noise scale matrix (F_param * I), empty if not augmented
    """
    nx = 6
    Nplus1 = xa.shape[1]

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
              + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)

    Xi_full = constants.Xi
    lib_cfg = constants.lib_cfg
    swat = constants.swat_ceiling
    xi_idx = constants.xi_idx
    F_dyn = constants.F_dyn
    F_param = constants.F_param

    x_state = xa[:nx, :]
    if augment_states:
        xi_vec = xa[nx:, :]
        n_p = xi_vec.shape[0]
        if n_p != xi_idx.shape[0]:
            raise ValueError(
                f"continuous_state_dynamics: n_p from xa ({n_p}) does not "
                f"match xi_idx rows ({xi_idx.shape[0]}). This usually means "
                "xa was double-augmented (iLQG appended p_hat to an "
                "already-augmented x0). Pass only x_hat (nx rows) as x0 to "
                "ilqg_function, not [x_hat; p_hat].")
    else:
        xi_vec = np.zeros((0, Nplus1))
        n_p = 0

    f = np.zeros((nx, Nplus1))
    for col in range(Nplus1):
        x_col = x_state[:, col]
        u_col = u[:, min(col, u.shape[1] - 1)]

        Xi_col = Xi_full.copy()
        for k in range(n_p):
            t_idx, s_idx = xi_idx[k, 0], xi_idx[k, 1]
            Xi_col[t_idx, s_idx] = xi_vec[k, col]

        theta_row = build_sindy_row(x_col, u_col, lib_cfg)
        f_col = theta_row @ Xi_col

        f_col[2:6] = 0.0  # x3..x6 (AIS, Age, DPI, Caregiver) are static

        for s in range(2):  # recovery floor/ceiling on SCIM, BBS only
            if x_col[s] <= 0 and f_col[s] < 0:
                f_col[s] = 0.0
            elif x_col[s] >= swat and f_col[s] > 0:
                f_col[s] = 0.0

        f[:, col] = f_col

    if augment_states:
        fp = np.zeros((n_p, Nplus1))  # parameter dynamics: zero-mean random walk
        xdot = np.vstack([f, fp])
    else:
        xdot = f

    F = F_dyn * np.eye(nx)
    Fp = F_param * np.eye(n_p) if augment_states else np.zeros((0, 0))

    return xdot, nx, n_p, F, Fp


def build_sindy_row(x, u_vec, lib_cfg):
    """
    One-row SINDy candidate-function evaluation: [1, x, u, x^2, u^2,
    cross-products, x1^3, x2^3, sin/cos] -- must match
    build_sindy_library.m's Theta column order exactly.

    NOTE: build_sindy_library.m, ContinuousStateDynamics.m, and
    validate_sindy_model.m each carry their OWN independent copy of this
    row-builder in the MATLAB source -- there is nothing enforcing that
    the three stay in sync. This is a latent fragility in the original
    code (see validate_sindy_model.py's module docstring for a concrete
    case: its MATLAB copy's cubic-term count uses a wrong-in-general
    approximation that happens to coincide with the correct value at
    Katie's actual dims). PUBLIC (no leading underscore) and imported
    directly by validate_sindy_model.py instead of being copy-pasted a
    third time on the Python side -- one shared implementation instead
    of MATLAB's three independent ones.

    lib_cfg: dict with 'poly_order' (int), 'include_cross' (bool),
    optionally 'include_trig' (bool, default False -- matches this
    file's defensive `isfield` check in the MATLAB source, which is
    stricter than build_sindy_library.m's own unchecked `cfg.include_trig`).
    """
    z = np.concatenate([np.asarray(x).ravel(), np.asarray(u_vec).ravel()])
    n_vars = z.shape[0]
    n_states = np.asarray(x).ravel().shape[0]

    cols = [1.0]
    cols.extend(z[i] for i in range(n_vars))

    poly_order = lib_cfg["poly_order"]
    if poly_order >= 2:
        cols.extend(z[i] ** 2 for i in range(n_vars))
        if lib_cfg.get("include_cross", False):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    cols.append(z[i] * z[j])
    if poly_order >= 3:
        for i in range(min(n_states, 2)):
            cols.append(z[i] ** 3)
    if lib_cfg.get("include_trig", False):
        for i in range(n_vars):
            cols.append(np.sin(z[i]))
            cols.append(np.cos(z[i]))

    return np.array(cols)
