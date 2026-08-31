"""
Observation model for duopoly pricing. Both realized demands are observed
directly each period (sales are just counted per seller), with small
per-seller reporting noise:
    y_A = x1 + sigma_y_A * v1 * sqrt(dt)
    y_B = x2 + sigma_y_B * v2 * sqrt(dt)

BOTH demands must be observed (not just "your own"): the cross-price
terms b_AB/b_BA are only identifiable from how the OTHER seller's demand
responds to a price move (dD_A/db_AB = p_B lives in D_A's residual, not
D_B's) -- see continuous_dynamics.py's docstring. This is a direct
2-seller generalization of applications/dynamic_pricing/measurement.py.

Matches MeasurementFn's contract exactly (extensions/dual_control/protocols.py):
    measurement(dt, xa, u, constants, v, noise_index, augment_states,
                v_from_filter) -> (ny, K) array
noise_index=False/0 -> the real (noisy) observation g(x) + G v sqrt(dt).
noise_index=i (1-indexed) -> the i-th column of the measurement-noise
diffusion matrix G, broadcast across all K columns -- comes out
identically zero here since G doesn't depend on x or u (same as
dynamic_pricing/measurement.py's G).

`u` and `constants`'s other fields are unused here -- the observation
depends only on x1/x2, never on price or the demand-curve parameters --
kept as unused parameters for signature fidelity with the shared
measurement() contract, same convention as dynamic_pricing/measurement.py's
unused `u`.
"""

import numpy as np


def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                  v_from_filter):
    Kplus1 = xa.shape[1]
    G = np.diag([constants.sigma_y_A, constants.sigma_y_B])  # (ny, nv) = (2, 2)

    if noise_index:  # 1-indexed request for the noise-diffusion column
        col = G[:, noise_index - 1]
        return np.tile(col[:, None], (1, Kplus1))

    g = xa[0:2, :]  # observe both realized demands directly

    if not v_from_filter:
        noise = (G @ v) * np.sqrt(dt)
    else:
        noise = v

    return g + noise
