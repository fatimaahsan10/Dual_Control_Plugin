"""
Observation model for SCI rehabilitation state estimation.

Ported from Katie's CODES/KATIE'S CODE/.../Measurement.m.

At each rehab session, only the two DYNAMIC states are observed:
    y = [SCIM_norm; BBS_norm] + G @ v * sqrt(dt)
(AIS, Age, DPI, Caregiver support are fixed patient attributes, known
at admission, never observed through a noisy channel here.) G is a
constant diagonal instrument-noise scaling -- SCIM has higher
inter-rater reliability (lower noise) than BBS.

Matches forward_pass.py's `measurement` callable contract exactly:
    measurement(dt, xa, u, constants, v, noise_index, augment_states,
                v_from_filter) -> (ny, K) array
noise_index=False -> the real (noisy) observation g(x) + G v sqrt(dt).
noise_index=i (1-indexed, matching MATLAB) -> the i-th column of the
measurement-noise diffusion matrix G, broadcast across all K columns --
this is what forward_pass.py differentiates to build Dx/Du, and comes
out identically zero there since G doesn't depend on x or u.

DEVIATION FROM THE LITERAL MATLAB SOURCE: `u`, `constants`, and
Measurement.m's broadcasting of `u` to match xk's column count are all
dead in the MATLAB source -- g depends only on xk, never on u or the
constants struct. Kept as unused parameters here for signature fidelity
with the shared measurement() contract (same convention as backward_pass.py's
unused `E`).
"""

import numpy as np

_G = np.diag([0.03, 0.05])  # [SCIM noise scale, BBS noise scale], (ny, nv)


def measurement(dt, xk, u, constants, v, noise_index, augment_states,
                  v_from_filter):
    Np1 = xk.shape[1]

    if noise_index:  # 1-indexed request for the noise-diffusion column
        col = _G[:, noise_index - 1]
        return np.tile(col[:, None], (1, Np1))

    g = xk[:2, :]  # observe SCIM (x1), BBS (x2) directly

    if not v_from_filter:
        noise = (_G @ v) * np.sqrt(dt)
    else:
        noise = v

    return g + noise
