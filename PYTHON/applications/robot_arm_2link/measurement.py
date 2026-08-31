"""
Observation model for the 2-link robot arm: joint encoders + tachometers
read q1, q2, w1, w2 directly each control step, with small per-channel
sensor noise:
    y1 = q1 + sigma_y_q * v1 * sqrt(dt)
    y2 = q2 + sigma_y_q * v2 * sqrt(dt)
    y3 = w1 + sigma_y_w * v3 * sqrt(dt)
    y4 = w2 + sigma_y_w * v4 * sqrt(dt)

m2 (the estimated parameter, xa row 4 when augmented) is NEVER directly
observed -- there's no scale on the arm. It's inferred purely through
how it shapes the observed q/w trajectory under applied torque, which is
the actual point of the dual-control exercise. Same
observe-the-physical-state-only convention as
applications/duopoly_pricing/measurement.py.

Matches MeasurementFn's contract exactly (extensions/dual_control/protocols.py):
    measurement(dt, xa, u, constants, v, noise_index, augment_states,
                v_from_filter) -> (ny, K) array
noise_index=False/0 -> the real (noisy) observation g(x) + G v sqrt(dt).
noise_index=i (1-indexed) -> the i-th column of the measurement-noise
diffusion matrix G, broadcast across all K columns -- comes out
identically zero here since G doesn't depend on x or u.

`u` is unused -- the observation depends only on the physical state,
never on applied torque -- kept as an unused parameter for signature
fidelity, same convention as duopoly_pricing/measurement.py's unused `u`.
"""

import numpy as np


def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                  v_from_filter):
    Kplus1 = xa.shape[1]
    G = np.diag([constants.sigma_y_q, constants.sigma_y_q,
                  constants.sigma_y_w, constants.sigma_y_w])  # (ny, nv) = (4, 4)

    if noise_index:  # 1-indexed request for the noise-diffusion column
        col = G[:, noise_index - 1]
        return np.tile(col[:, None], (1, Kplus1))

    g = xa[0:4, :]  # observe q1, q2, w1, w2 directly

    if not v_from_filter:
        noise = (G @ v) * np.sqrt(dt)
    else:
        noise = v

    return g + noise
