"""
Confirms simulate_system.py -- ported from Andrew's
CODES/ANDREW'S CODE/simulate_system.m -- works unchanged at Katie's
dimensions (6 states, 13 actions), same approach as the other
test_*_dims.py files in this directory.

Diff check (Compare-Object, Andrew's vs Katie's simulate_system.m):
Katie's version only drops the `if ~isreal(x_ode) warning(...)` check,
renames a couple of local variables (x_ode -> x_next), and adds a
header doc-comment -- the propagation loop itself (extend U by
repeating the last control, then alternate dynamics/measurement calls)
is unchanged. Matches CLAUDE.md's "trivial diff" characterization.

simulate_system.py was only exercised transitively before (inside
ilqg_function.py's rollout, via andrew/test_ilqg_function.py at
nx=2/nu=1) -- no standalone test existed for it even at Andrew's own
dimensions. It never hardcodes a dimension (nx/nv/nw/ny are all read
off xa0/sqrtR/sqrtQ/measurement-output shapes), so this test drives it
directly at nx=6/nu=13 with deterministic (zero-noise) linear
dynamics/measurement and checks the rollout against an independently
computed manual iteration -- catching indexing/transpose bugs
regardless of dimension.
"""

import numpy as np

from extensions.dual_control.simulate_system import simulate_system

# Katie's problem dimensions (SCIM, BBS, AIS, Age, DPI, Caregiver support;
# 13-therapy WHO Package of Interventions action space).
N_STATES = 6
N_ACTIONS = 13


def test_simulate_system_matches_manual_rollout_at_6state_13action():
    rng = np.random.default_rng(0)
    nx, nu, ny, M = N_STATES, N_ACTIONS, N_STATES, 5

    A = 0.9 * np.eye(nx) + 0.05 * rng.standard_normal((nx, nx))
    B = 0.1 * rng.standard_normal((nx, nu))
    F = 0.3 * rng.standard_normal((ny, nx))
    E = 0.2 * rng.standard_normal((ny, nu))

    def dynamics(dt, xa, u, constants, w, noise_index, augment_states, w_from_filter,
                  lam, u_lims, u_lim_method):
        return A @ xa + B @ u + w  # w is 0 here (sqrtQ is all-zero below)

    def measurement(dt, xa, u, constants, v, noise_index, augment_states, v_from_filter):
        return F @ xa + E @ u + v  # v is 0 here (sqrtR is all-zero below)

    xa0 = rng.standard_normal(nx)
    U = rng.standard_normal((nu, M))
    sqrtR = np.zeros((ny, ny))  # deterministic: zero measurement noise
    sqrtQ = np.zeros((nx, nx))  # deterministic: zero process noise

    y, xa = simulate_system(
        dt=1.0, xa0=xa0, U=U, constants=None, sqrtR=sqrtR, sqrtQ=sqrtQ,
        augment_states=True, lam=0.0, u_lims=None, u_lim_method=None,
        dynamics=dynamics, measurement=measurement)

    assert xa.shape == (nx, M + 1)
    assert y.shape == (ny, M + 1)

    # Independent manual rollout: U's last column is repeated once
    # (matches MATLAB's `U(:, end+1) = U(:, end)`), so there are M
    # dynamics steps total, using U[:, i] for i = 0..M-1 unchanged.
    xa_ref = np.zeros((nx, M + 1))
    y_ref = np.zeros((ny, M + 1))
    xa_ref[:, 0] = xa0
    y_ref[:, 0] = F @ xa0 + E @ U[:, 0]
    for i in range(M):
        xa_ref[:, i + 1] = A @ xa_ref[:, i] + B @ U[:, i]
        u_for_measurement = U[:, i + 1] if i + 1 < M else U[:, -1]
        y_ref[:, i + 1] = F @ xa_ref[:, i + 1] + E @ u_for_measurement

    assert np.allclose(xa, xa_ref, atol=1e-10)
    assert np.allclose(y, y_ref, atol=1e-10)


if __name__ == "__main__":
    tests = [test_simulate_system_matches_manual_rollout_at_6state_13action]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")
