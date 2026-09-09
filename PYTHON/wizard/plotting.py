"""
Turns a main_outer_control_loop() result dict (see
extensions/dual_control/main_outer_control_loop.py's own docstring for its
exact keys/shapes) into matplotlib figures and a results table -- the
wizard's "Run and display results" step.

Every function here defends against the inputs it's actually given
possibly being degenerate (empty arrays, all-NaN columns that slipped
past solver_runner.py's own NaN/Inf sweep in some untested corner, zero
parameters) by checking shapes/finiteness before handing anything to
matplotlib, and returns None (never raises) when there's nothing sensible
to plot -- matching the wizard's "a failure in one step must not crash
the session" requirement even at the very last, purely-cosmetic step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from wizard.schema import ModelConfig


def _has_finite_data(arr, ndim: int) -> bool:
    """Guards every plot function's entry point: `arr` must actually be an
    ndarray of the expected dimensionality with at least one finite value,
    or there's nothing safe to index into/plot. Catches malformed results
    (wrong shape, wrong type, all-NaN) uniformly instead of letting a
    downstream `.shape[1]` on a 1-D array (or similar) raise."""
    return (isinstance(arr, np.ndarray) and arr.ndim == ndim
            and arr.size > 0 and np.isfinite(arr).any())


def plot_states(result: dict, config: ModelConfig) -> Figure | None:
    x_true = result.get("x_true")
    x_hat = result.get("x_hat")
    if not _has_finite_data(x_true, ndim=2):
        return None

    nx = x_true.shape[0]
    N = x_true.shape[1]
    sessions = np.arange(N)

    fig = Figure(figsize=(7, 2.2 * nx))
    for i, state in enumerate(config.states):
        ax = fig.add_subplot(nx, 1, i + 1)
        ax.plot(sessions, x_true[i, :], label="true", marker="o", markersize=3)
        if isinstance(x_hat, np.ndarray) and x_hat.ndim == 2 and x_hat.shape[0] > i:
            ax.plot(sessions[:x_hat.shape[1]], x_hat[i, :], label="estimated",
                     linestyle="--")
        ax.set_ylabel(state.label or state.name)
        ax.set_xlabel("step")
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def plot_actions(result: dict, config: ModelConfig) -> Figure | None:
    u = result.get("u")
    if not _has_finite_data(u, ndim=2):
        return None

    nu = u.shape[0]
    steps = np.arange(u.shape[1])

    fig = Figure(figsize=(7, 2.0 * nu))
    for j, action in enumerate(config.actions):
        ax = fig.add_subplot(nu, 1, j + 1)
        ax.step(steps, u[j, :], where="post")
        ax.axhline(action.min, color="gray", linestyle=":", linewidth=1)
        ax.axhline(action.max, color="gray", linestyle=":", linewidth=1)
        ax.set_ylabel(action.label or action.name)
        ax.set_xlabel("step")
    fig.tight_layout()
    return fig


def plot_parameters(result: dict, config: ModelConfig) -> Figure | None:
    if config.n_p == 0:
        return None
    p_hat = result.get("p_hat")
    if not _has_finite_data(p_hat, ndim=2):
        return None

    n_p = p_hat.shape[0]
    sessions = np.arange(p_hat.shape[1])

    fig = Figure(figsize=(7, 2.0 * n_p))
    for k, param in enumerate(config.parameters):
        ax = fig.add_subplot(n_p, 1, k + 1)
        ax.plot(sessions, p_hat[k, :], label="estimate", marker="o", markersize=3)
        if param.true_value is not None:
            ax.axhline(param.true_value, color="green", linestyle="--",
                        label="true value")
        ax.axhline(param.prior_guess, color="gray", linestyle=":",
                    label="initial guess")
        ax.set_ylabel(param.label or param.name)
        ax.set_xlabel("step")
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def plot_convergence(result: dict) -> Figure | None:
    """iLQR-only diagnostic (control_method == "ilqr", see schema.py) --
    core.ddp_solver.ilqg.ilqg()'s own `trace` (one dict per iteration,
    each carrying at least "cost" and "grad_norm" once a backward pass
    succeeds -- see ilqg.py's docstring) has no equivalent for the iLQG/
    dual-control path (there is no single "convergence" to plot for an
    MPC loop that replans every step), so this is never called from that
    path."""
    trace = result.get("trace")
    if not isinstance(trace, list) or not trace:
        return None

    iters = [r["iter"] for r in trace if "cost" in r]
    costs = [r["cost"] for r in trace if "cost" in r]
    grad_iters = [r["iter"] for r in trace if "grad_norm" in r]
    grad_norms = [r["grad_norm"] for r in trace if "grad_norm" in r]
    if not costs and not grad_norms:
        return None

    fig = Figure(figsize=(7, 4.0))
    ax1 = fig.add_subplot(2, 1, 1)
    if costs:
        ax1.plot(iters, costs, marker="o", markersize=3)
    ax1.set_ylabel("total cost")
    ax1.set_xlabel("iteration")
    ax1.set_title("iLQR convergence")

    ax2 = fig.add_subplot(2, 1, 2)
    if grad_norms:
        ax2.plot(grad_iters, grad_norms, marker="o", markersize=3, color="tab:orange")
        ax2.set_yscale("log")
    ax2.set_ylabel("gradient norm")
    ax2.set_xlabel("iteration")
    fig.tight_layout()
    return fig


def plot_cost(result: dict) -> Figure | None:
    cost_true = result.get("cost_true")
    if not _has_finite_data(cost_true, ndim=1):
        return None

    steps = np.arange(cost_true.shape[0])
    cumulative = np.cumsum(cost_true)

    fig = Figure(figsize=(7, 4.0))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(steps, cost_true, marker="o", markersize=3)
    ax1.set_ylabel("cost per step")
    ax1.set_xlabel("step")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(steps, cumulative, marker="o", markersize=3, color="tab:orange")
    ax2.set_ylabel("cumulative cost")
    ax2.set_xlabel("step")
    fig.tight_layout()
    return fig


def results_table(result: dict, config: ModelConfig) -> pd.DataFrame | None:
    """
    One row per simulated step. Columns are named after the model's own
    state/action/parameter names -- never generic 'x1'/'u1' placeholders
    unless that's literally what the user named them.
    """
    x_true = result.get("x_true")
    if not _has_finite_data(x_true, ndim=2):
        return None

    N = x_true.shape[1]
    data: dict[str, np.ndarray] = {"step": np.arange(N)}

    for i, state in enumerate(config.states):
        data[f"{state.name} (true)"] = x_true[i, :]

    def _add_row_column(name: str, row: np.ndarray) -> None:
        col = np.full(N, np.nan)
        m = min(N, row.shape[0])
        col[:m] = row[:m]
        data[name] = col

    x_hat = result.get("x_hat")
    if isinstance(x_hat, np.ndarray) and x_hat.ndim == 2:
        for i, state in enumerate(config.states):
            if i < x_hat.shape[0]:
                _add_row_column(f"{state.name} (estimated)", x_hat[i, :])

    u = result.get("u")
    if isinstance(u, np.ndarray) and u.ndim == 2:
        for j, action in enumerate(config.actions):
            if j < u.shape[0]:
                _add_row_column(action.name, u[j, :])

    p_hat = result.get("p_hat")
    if isinstance(p_hat, np.ndarray) and p_hat.ndim == 2:
        for k, param in enumerate(config.parameters):
            if k < p_hat.shape[0]:
                _add_row_column(f"{param.name} (estimated)", p_hat[k, :])

    for key in ("cost_true", "cost_est"):
        arr = result.get(key)
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            _add_row_column(key, arr)

    return pd.DataFrame(data)
