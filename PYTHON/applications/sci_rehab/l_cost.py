"""
Running and terminal cost for SCI rehabilitation planning.

Ported from Katie's CODES/KATIE'S CODE/.../l_cost.m.

    cost = state cost + resource cost ( + terminal cost at horizon end )
      state    : weighted squared gap of SCIM and BBS from the
                 persona-specific recovery ceiling (swat) -- drives both
                 toward swat
      resource : ONE small uniform penalty on every therapy dose
                 (w_resource * sum_j u_j^2) -- "doing anything costs
                 something" (replaces the older, therapy-specific
                 w_intensity/w_frequency weights from the 3-control demo)
      terminal : same gap, larger persona-specific weight, applied only
                 at the final step

    xa: xa[0]=SCIM  xa[1]=BBS  xa[2]=AIS  xa[3]=Age  xa[4]=DPI  xa[5]=Caregiver
    u : 13 WHO therapy doses U1..U13 (who_therapies.m), each in [0, 1]

All cost weights and the recovery ceiling are persona-specific, read
from `constants` (see sindy_model_constants.py) instead of MATLAB's
`global SINDY_MODEL`; see define_persona.m for where they originate.

Matches forward_pass.py's `cost` callable contract exactly:
    cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants)
    -> (K,) array. u columns may be all-NaN to signal "final cost only";
    cost() zeroes them before use.

DEVIATION FROM THE LITERAL MATLAB SOURCE: `tracking_trajectory` is
entirely unused in l_cost.m -- this cost tracks a fixed ceiling, not a
time-varying reference trajectory -- kept as an unused parameter for
signature fidelity with the shared cost() contract, same convention as
Measurement.m's unused `u`/`constants`.
"""

import numpy as np

from applications.sci_rehab.sindy_model_constants import PersonaWeights

_DEFAULT_WEIGHTS = PersonaWeights(
    w_scim=2.0, w_bbs=1.0, w_resource=1e-2,
    w_terminal_scim=50.0, w_terminal_bbs=25.0)


def l_cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
    # Final-step controls arrive as NaN -- zero them so they add no
    # resource cost. Copied (not mutated in place) since Python arrays
    # are references, unlike MATLAB's pass-by-value semantics: the
    # caller's `u` (e.g. forward_pass.py's u_aug) must not be clobbered.
    u = np.array(u, dtype=float, copy=True)
    final = np.isnan(u[0, :])
    u[:, final] = 0.0

    if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
        u_phys = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
                    + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)
    else:
        u_phys = u

    swat = constants.swat_ceiling if constants is not None else 1.0
    w = (constants.persona
         if constants is not None and constants.persona is not None
         else _DEFAULT_WEIGHTS)

    gap_scim = swat - xa[0, :]
    gap_bbs = swat - xa[1, :]

    # tiny 1e-6*sum(u^2) term keeps the iLQG Hessian well-conditioned
    # where tanh saturates -- negligible against the real costs.
    lx = w.w_scim * gap_scim ** 2 + w.w_bbs * gap_bbs ** 2
    lu = w.w_resource * np.sum(u_phys ** 2, axis=0) + 1e-6 * np.sum(u ** 2, axis=0)

    lf = np.zeros(xa.shape[1])
    lf[final] = (w.w_terminal_scim * gap_scim[final] ** 2
                  + w.w_terminal_bbs * gap_bbs[final] ** 2)

    return lx + lu + lf
