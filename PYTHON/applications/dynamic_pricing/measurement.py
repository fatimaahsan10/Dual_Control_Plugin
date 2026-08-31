"""
Observation model for dynamic pricing.

Realized demand is observed directly each period (sales are just
counted), with small reporting noise:
    y = x1 + sigma_y * v * sqrt(dt)

Matches MeasurementFn's contract exactly (extensions/dual_control/protocols.py):
    measurement(dt, xa, u, constants, v, noise_index, augment_states,
                v_from_filter) -> (ny, K) array
noise_index=False/0 -> the real (noisy) observation g(x) + G v sqrt(dt).
noise_index=i (1-indexed) -> the i-th column of the measurement-noise
diffusion matrix G, broadcast across all K columns -- comes out
identically zero here since G doesn't depend on x or u (same as
applications/sci_rehab/measurement.py's _G).

`u` and `constants`'s other fields are unused here -- the observation
depends only on x1, never on price or the demand-curve parameters --
kept as unused parameters for signature fidelity with the shared
measurement() contract, same convention as applications/sci_rehab/
measurement.py's unused `u`/`constants`.
"""

import numpy as np


def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                  v_from_filter):
    Kplus1 = xa.shape[1]
    G = np.array([[constants.sigma_y]])  # (ny, nv) = (1, 1)

    if noise_index:  # 1-indexed request for the noise-diffusion column
        col = G[:, noise_index - 1]
        return np.tile(col[:, None], (1, Kplus1))

    g = xa[0:1, :]  # observe realized demand x1 directly

    if not v_from_filter:
        noise = (G @ v) * np.sqrt(dt)
    else:
        noise = v

    return g + noise
