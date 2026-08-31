"""
Reduces a state-only inequality constraint h(x) >= 0 into a
control-dependent one h_tilde(x, u) >= 0, so it can be handed straight to
dynamic_control_bounds.py's Variant-A machinery.

Implements Variant B of Dastan & Sensinger (2024), "Leveraging Control
Inputs to Enforce Constraints in Differential Dynamic Programming for
Nonlinear Optimization" (IEEE CDC 2024, DOI 10.1109/CDC56724.2024.10886170)
-- eqs (15)-(19).

    h_dot(x)   = dh/dx . f(x,u)                (eq 16-17, Lie derivative
                                                 of h along the CONTINUOUS
                                                 dynamics x_dot = f(x,u))
    h_tilde    = h_dot(x) + alpha * h(x)        (eq 18-19, alpha > 0)

Assumes relative degree one: h_dot must actually depend on u (i.e. f's
control-channel doesn't cancel out of dh/dx . f). dynamic_control_bounds.py
surfaces this as a RuntimeWarning (not a silent no-op) if it turns out not
to hold at a given (x, u).

Same "no source code exists for this paper" caveat as
dynamic_control_bounds.py -- original implementation from the paper's
equations, verified against its own worked pendulum example (Sec IV-A).
"""

import numpy as np

from extensions.dual_control.finite_difference import finite_difference


def lie_derivative_h(h_fn, dynamics_fn, x, u, dhdx_fn=None, fd_step=2 ** -17):
    """
    Paper eq (16)-(17): h_dot(x) = dh/dx . f(x,u).

    Parameters
    ----------
    h_fn        : callable(x) -> (p,) ndarray. State-only constraint,
                  no dependence on u.
    dynamics_fn : callable(x, u) -> (n,) ndarray. CONTINUOUS-time vector
                  field x_dot = f(x, u) (paper eq 1) -- NOT a discrete
                  one-step map like todorov's step_fn/ilqg.py dynamics
                  callable; this reduction is a continuous-time
                  construction (the chain rule in eq 16 differentiates
                  through x_dot).
    x, u        : (n,), (m,)
    dhdx_fn     : callable(x) -> (p, n) ndarray, optional analytic
                  Jacobian dh/dx. If None, computed via forward
                  differences (andrew/finite_difference.py).
    fd_step     : float, forward-difference step when dhdx_fn is None.

    Returns
    -------
    (p,) ndarray -- h_dot(x) at this (x, u).
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    n = x.shape[0]

    if dhdx_fn is not None:
        J = np.atleast_2d(np.asarray(dhdx_fn(x), dtype=float))
    else:
        p = np.atleast_1d(np.asarray(h_fn(x), dtype=float)).shape[0]

        def h_of_x(X):  # (n, K) -> (p, K)
            K = X.shape[1]
            out = np.empty((p, K))
            for k in range(K):
                out[:, k] = h_fn(X[:, k])
            return out

        J = finite_difference(h_of_x, x.reshape(n, 1), h=fd_step)[:, :, 0]  # (p, n)

    f_val = np.asarray(dynamics_fn(x, u), dtype=float)
    return J @ f_val


def state_constraint_to_control_constraint(h_fn, dynamics_fn, alpha,
                                             dhdx_fn=None, fd_step=2 ** -17):
    """
    Paper eq (18)-(19): builds h_tilde(x,u) = h_dot(x) + alpha*h(x),
    directly usable as the `h_fn` argument to
    dynamic_control_bounds.solve_box_from_constraint /
    build_time_varying_lims.

    Parameters
    ----------
    h_fn, dynamics_fn, dhdx_fn, fd_step : see lie_derivative_h.
    alpha : float > 0. Paper: "a constant that can be determined through
            iterations, ensuring that the constraint is not violated" --
            no closed-form selection rule is given (e.g. alpha=0.1 for
            the pendulum, but three DIFFERENT values for the 2D car's
            three obstacles), so this is a required, explicit argument
            rather than a default.

    Returns
    -------
    h_tilde_fn : callable(x, u) -> (p,) ndarray
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0 (paper eq 18), got {alpha}")

    def h_tilde_fn(x, u):
        h_val = np.atleast_1d(np.asarray(h_fn(x), dtype=float))
        h_dot = lie_derivative_h(h_fn, dynamics_fn, x, u, dhdx_fn, fd_step)
        return h_dot + alpha * h_val

    return h_tilde_fn
