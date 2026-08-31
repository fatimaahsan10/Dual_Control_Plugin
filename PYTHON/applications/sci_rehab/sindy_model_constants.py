"""
Explicit substitute for MATLAB's `global SINDY_MODEL`, which
ContinuousStateDynamics.m and l_cost.m (via get_cost_params()) both read
directly instead of using their own function arguments.

Ported alongside continuous_state_dynamics.py / l_cost.py.

DEVIATION FROM THE LITERAL MATLAB SOURCE: rather than reaching into a
Python global, continuous_state_dynamics.py and l_cost.py take an
instance of SindyModelConstants as their `constants` argument -- the
same slot forward_pass.py / simulate_system.py / ilqg_function.py
already thread through (unused, for Andrew's SIDARTHE plant). Same
"make the hardcoded/global plant dependency explicit" convention used
throughout this port (see todorov_estimator.py's `measurement`
parameter, forward_pass.py's `dynamics`/`measurement`/`cost`
parameters).

NOTE: ilqg_function.py's augment_states=False branch does
`constants = np.concatenate([constants, p_hat])`, which requires
`constants` to be a numpy array -- SindyModelConstants is NOT
compatible with that branch. Katie's plant always runs with
augment_states=True (SINDy coefficients are estimated online via dual
iLQG -- see ContinuousStateDynamics.m's own docstring), so that branch
is never exercised in practice. Flagged here since it's a real
constraint, not something silently handled.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PersonaWeights:
    """Mirrors define_persona.m's cost-weight fields."""
    w_scim: float
    w_bbs: float
    w_resource: float
    w_terminal_scim: float
    w_terminal_bbs: float


@dataclass
class SindyModelConstants:
    """
    Mirrors the MATLAB `global SINDY_MODEL` struct's fields, as read by
    ContinuousStateDynamics.m and l_cost.m's local get_cost_params().

    Xi           : (n_terms, nx) full SINDy coefficient matrix (physical units)
    lib_cfg      : dict with keys 'poly_order' (int), 'include_cross' (bool),
                   optionally 'include_trig' (bool, default False) -- matches
                   build_sindy_library.m's cfg struct
    swat_ceiling : float, patient-specific recovery ceiling in [0, 1]
    xi_idx       : (n_p, 2) int array of [term_idx, state_idx], 0-INDEXED
                   here (MATLAB source is 1-indexed) -- coefficients being
                   estimated online when augment_states=True
    F_dyn        : float, dynamics noise scale (MATLAB default 1e-3)
    F_param      : float, parameter noise scale (MATLAB default 1e-4)
    persona      : PersonaWeights or None -- None falls back to
                   l_cost.py's population-average defaults, same as MATLAB
                   when SINDY_MODEL.persona is missing
    """
    Xi: np.ndarray
    lib_cfg: dict
    swat_ceiling: float
    xi_idx: np.ndarray
    F_dyn: float = 1e-3
    F_param: float = 1e-4
    persona: Optional[PersonaWeights] = None
