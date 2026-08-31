"""
The one generic factory that turns a validated ModelConfig (schema.py)
into the four callables extensions/dual_control/protocols.py's
DynamicsFn/MeasurementFn/CostFn/ContinuousDynamicsFn describe -- built
ONCE, generically, parameterized by dims and by the lambdified per-
equation functions equation_parser.py produces, rather than by
generating and exec'ing per-model Python source text (see wizard's
design plan: codegen_preview.py separately renders human-readable source
for display, but the app itself always RUNS these closures, never
`exec`'d text).

Confirmed by diffing applications/dynamic_pricing/ against
applications/duopoly_pricing/ before writing this: dynamics.py's Euler-
Maruyama step, the noise_index/w_from_filter branches, and the NaN-
terminal-column handling in cost.py are STRUCTURALLY IDENTICAL between a
1-state/1-unknown-parameter plant and a 2-state/4-unknown-parameter plant
with cross-terms -- only the per-equation arithmetic and the dims differ.
That's exactly what's parameterized here.

CALLING CONVENTION for every lambdified equation this module calls: each
takes positional args in the FIXED order
    (state_1, ..., state_nx, action_1, ..., action_nu, param_1, ..., param_n_p)
-- see compile_plant()'s `ordered_symbols` -- regardless of which of those
names actually appear in a given equation (sympy.lambdify happily accepts
and ignores unused parameters). Every state/action row and resolved
parameter row is always a 1-D array of length K (the batch size), so plain
elementwise numpy arithmetic inside the lambdified body vectorizes across
K automatically, matching every hand-written plant file in this codebase.

`constants` (the object threaded through the DynamicsFn/CostFn/etc.
contract) is DELIBERATELY MINIMAL here -- a GenericConstants holding only
`prior` (the unknown-parameter fallback values used when
`augment_states=False`, exactly mirroring dynamic_pricing/duopoly_pricing's
own `constants.b_prior` fallback convention). Known constants (the
config's ConstantSpec values) are NOT stored on `constants` at all --
they're substituted directly into each equation's sympy expression at
COMPILE time (compile_plant(), via `.subs(...)`), before lambdify, since
they never change during a run. This is a deliberate simplification
relative to the hand-written examples (which keep economic constants on
a per-plant dataclass for readability) -- functionally equivalent, since
protocols.py documents `constants: Any` as entirely plant-defined and
never inspected by this layer.

Zero changes to core/ddp_solver/protocols.py or
extensions/dual_control/protocols.py -- this module only produces
callables that satisfy those (unmodified) Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import sympy

from wizard.equation_parser import parse_equation, lambdify_equation
from wizard.schema import ModelConfig


class PlantCompilationError(Exception):
    """
    Raised by compile_plant() when one or more equations in the config
    fail to parse. Carries EVERY equation error found (not just the
    first), each already prefixed with which equation it came from (e.g.
    "Equation for state 'x1': ...") -- same "report everything at once"
    convention as core/ddp_solver/plugin_validation.py's
    PluginContractError.
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class GenericConstants:
    """The only `constants` object generic_plant.py ever builds -- see
    module docstring for why known constants aren't stored here."""
    prior: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class CompiledPlant:
    dynamics: Callable
    measurement: Callable
    cost: Callable
    continuous_dynamics: Callable
    constants: GenericConstants
    nx: int
    nu: int
    n_p: int
    ny: int


def _block_diag(*mats: np.ndarray) -> np.ndarray:
    mats = [np.atleast_2d(m) for m in mats]
    total_rows = sum(m.shape[0] for m in mats)
    total_cols = sum(m.shape[1] for m in mats)
    out = np.zeros((total_rows, total_cols))
    i = j = 0
    for m in mats:
        r, c = m.shape
        out[i:i + r, j:j + c] = m
        i += r
        j += c
    return out


def _row(value, K: int) -> np.ndarray:
    """Broadcasts a lambdified equation's return value to a (K,) row.

    A parsed equation that reduces to a pure constant (e.g. a terminal
    cost of "0", or a parameter's own trivial dynamics) makes sympy.lambdify
    return a bare Python/numpy scalar regardless of its array inputs --
    confirmed directly while building this (a constant expression's
    generated code just does `return 0`, ignoring its arguments entirely).
    Every caller in this module needs a uniform (K,) array to np.vstack
    or combine with other rows, so every lambdified call is routed through
    this helper rather than assumed to already be array-shaped.
    """
    return np.broadcast_to(np.asarray(value, dtype=float), (K,)).copy()


def _resolve_param_rows(xa: np.ndarray, nx: int, n_p: int, K: int,
                          augment_states: bool, constants: GenericConstants
                          ) -> list[np.ndarray]:
    if n_p == 0:
        return []
    if augment_states:
        return [xa[nx + k, :] for k in range(n_p)]
    return [np.full(K, constants.prior[k]) for k in range(n_p)]


def make_continuous_dynamics_fn(nx: int, nu: int, n_p: int,
                                   xdot_fns: list[Callable],
                                   f_diag: list[float], fp_diag: list[float]
                                   ) -> Callable:
    """
    Builds `continuous_dynamics(dt, xa, u, constants, augment_states,
    u_lims, u_lim_method) -> (xdot, nx, n_p, F, Fp)` -- see
    ContinuousDynamicsFn in extensions/dual_control/protocols.py.

    Unknown parameters are modeled as a zero-drift random walk (`fp =
    zeros`) -- a fixed assumption matching BOTH real example plants
    (dynamic_pricing/, duopoly_pricing/) exactly, not something the
    wizard exposes as a per-model choice (there is no evidence either
    proven case, or any other plant in this codebase, ever needed
    anything else).
    """

    def continuous_dynamics(dt, xa, u, constants, augment_states, u_lims,
                              u_lim_method):
        K = xa.shape[1]
        if u.shape[1] == 1 and K > 1:
            u = np.tile(u, (1, K))
        if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
            u = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
                  + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)

        x_rows = [xa[i, :] for i in range(nx)]
        u_rows = [u[j, :] for j in range(nu)]
        p_rows = _resolve_param_rows(xa, nx, n_p, K, augment_states, constants)
        n_p_ret = n_p if augment_states else 0

        args = x_rows + u_rows + p_rows
        f_rows = [_row(fn(*args), K) for fn in xdot_fns]
        f = np.vstack(f_rows) if f_rows else np.zeros((0, K))

        if augment_states and n_p > 0:
            fp = np.zeros((n_p, K))
            xdot = np.vstack([f, fp])
        else:
            xdot = f

        F = np.diag(f_diag) if nx > 0 else np.zeros((0, 0))
        Fp = (np.diag(fp_diag) if (augment_states and n_p > 0)
               else np.zeros((0, 0)))

        return xdot, nx, n_p_ret, F, Fp

    return continuous_dynamics


def make_dynamics_fn(continuous_dynamics_fn: Callable,
                       state_clamps: dict[int, tuple[Optional[float], Optional[float]]]
                       ) -> Callable:
    """
    Builds `dynamics(dt, xa, u, constants, w, noise_index, augment_states,
    w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method) -> (nxa, K)`
    -- Euler-Maruyama step over `continuous_dynamics_fn`, mechanically
    identical to every hand-written dynamics.py in this codebase.
    `dyn_noise_reg_lambda` is accepted for signature fidelity but unused,
    same convention as every existing plant.
    """

    def dynamics(dt, xa, u, constants, w, noise_index, augment_states,
                  w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method):
        Kplus1 = xa.shape[1]
        if w.shape[1] == 1 and Kplus1 > 1:
            w = np.tile(w, (1, Kplus1))
        if u.shape[1] == 1 and Kplus1 > 1:
            u = np.tile(u, (1, Kplus1))

        fa, nx, n_p, F, Fp = continuous_dynamics_fn(
            dt, xa, u, constants, augment_states, u_lims, u_lim_method)
        F_a = _block_diag(F, Fp)

        if noise_index:  # 1-indexed request for the noise-diffusion column
            col = F_a[:, noise_index - 1]
            return np.tile(col[:, None], (1, Kplus1))

        if not w_from_filter:
            noise = (F_a @ w) * np.sqrt(dt)
        else:
            noise = w

        x_next = xa + fa * dt + noise
        for i, (lo, hi) in state_clamps.items():
            if lo is not None:
                x_next[i, :] = np.maximum(x_next[i, :], lo)
            if hi is not None:
                x_next[i, :] = np.minimum(x_next[i, :], hi)
        return x_next

    return dynamics


def make_measurement_fn(nx: int, nu: int, n_p: int, ny: int,
                          meas_fns: list[Callable], noise_scales: list[float]
                          ) -> Callable:
    """
    Builds `measurement(dt, xa, u, constants, v, noise_index,
    augment_states, v_from_filter) -> (ny, K)` -- see MeasurementFn in
    extensions/dual_control/protocols.py. Measurement expressions may
    reference states, actions, and/or unknown parameters (resolved the
    same augmented-vs-prior way as dynamics), even though neither proven
    example plant's measurement ever reads anything but states directly
    -- kept general since the protocol itself permits it.
    """

    def measurement(dt, xa, u, constants, v, noise_index, augment_states,
                      v_from_filter):
        Kplus1 = xa.shape[1]
        if u.shape[1] == 1 and Kplus1 > 1:
            u = np.tile(u, (1, Kplus1))

        G = np.diag(noise_scales) if ny > 0 else np.zeros((0, 0))

        if noise_index:  # 1-indexed request for the noise-diffusion column
            col = G[:, noise_index - 1]
            return np.tile(col[:, None], (1, Kplus1))

        x_rows = [xa[i, :] for i in range(nx)]
        u_rows = [u[j, :] for j in range(nu)]
        p_rows = _resolve_param_rows(xa, nx, n_p, Kplus1, augment_states, constants)
        args = x_rows + u_rows + p_rows

        g_rows = [_row(fn(*args), Kplus1) for fn in meas_fns]
        g = np.vstack(g_rows) if g_rows else np.zeros((0, Kplus1))

        if not v_from_filter:
            noise = (G @ v) * np.sqrt(dt)
        else:
            noise = v
        return g + noise

    return measurement


def make_cost_fn(nx: int, nu: int, n_p: int, running_fn: Callable,
                   terminal_fn: Callable) -> Callable:
    """
    Builds `cost(xa, u, tracking_trajectory, u_lims, u_lim_method,
    constants) -> (K,)` -- see CostFn in
    extensions/dual_control/protocols.py. `u` columns marked all-NaN
    (the documented "final step" signal) are zeroed BEFORE evaluating
    either cost expression (matching every existing plant's convention --
    the single most common CostFn bug per validate_plugin.py's own
    docstring), and the TERMINAL expression, not the running one, is used
    at those columns -- a strict generalization of the existing plants'
    hardcoded `total[final] = 0.0` (recovered exactly when the wizard's
    terminal-cost field is left at its default "0").
    """

    def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
        u = np.array(u, dtype=float, copy=True)
        final = np.isnan(u[0, :])
        u[:, final] = 0.0

        if u_lim_method == 2 and u_lims is not None and np.size(u_lims) > 0:
            u_phys = ((u_lims[:, 1] - u_lims[:, 0])[:, None] / 2 * np.tanh(u)
                        + (u_lims[:, 1] + u_lims[:, 0])[:, None] / 2)
        else:
            u_phys = u

        K = xa.shape[1]
        augment_states = xa.shape[0] > nx

        x_rows = [xa[i, :] for i in range(nx)]
        u_rows = [u_phys[j, :] for j in range(nu)]
        p_rows = _resolve_param_rows(xa, nx, n_p, K, augment_states, constants)
        args = x_rows + u_rows + p_rows

        running = _row(running_fn(*args), K)
        terminal = _row(terminal_fn(*args), K)
        return np.where(final, terminal, running)

    return cost


def _compile_equation(text: str, all_names: list[str], ordered_symbols: list[str],
                        const_values: dict[str, float], label: str,
                        errors: list[str]) -> Optional[Callable]:
    result = parse_equation(text, all_names)
    if not result.ok:
        errors.append(f"{label}: {result.error}")
        return None
    try:
        substituted = result.expr.subs(
            {sympy.Symbol(k): v for k, v in const_values.items()})
        return lambdify_equation(substituted, ordered_symbols)
    except Exception as e:  # noqa: BLE001 -- report, never propagate
        errors.append(f"{label}: could not compile this equation "
                       f"({type(e).__name__}: {e}).")
        return None


def compile_plant(config: ModelConfig) -> CompiledPlant:
    """
    The single entry point: validates `config` structurally (see
    schema.py's ModelConfig.validate()), parses every equation, and
    returns a CompiledPlant. Raises PlantCompilationError (never a bare
    sympy/numpy exception) listing every problem found if the config is
    structurally invalid OR any equation fails to parse -- a caller
    (validation_runner.py / the wizard UI) always gets one clear report,
    never a partial result or a crash.
    """
    struct_errors = config.validate()
    if struct_errors:
        raise PlantCompilationError(struct_errors)

    state_names = config.state_names()
    action_names = config.action_names()
    param_names = config.parameter_names()
    constant_names = config.constant_names()
    all_names = state_names + action_names + param_names + constant_names
    const_values = {c.name: c.value for c in config.constants}
    ordered_symbols = state_names + action_names + param_names

    errors: list[str] = []

    xdot_fns = []
    for s in config.states:
        fn = _compile_equation(config.dynamics.get(s.name, ""), all_names,
                                 ordered_symbols, const_values,
                                 f"Equation for state '{s.name}'", errors)
        xdot_fns.append(fn)

    meas_fns = []
    noise_scales = []
    for m in config.measurement:
        fn = _compile_equation(m.expression, all_names, ordered_symbols,
                                 const_values, f"Measured output '{m.name}'",
                                 errors)
        meas_fns.append(fn)
        noise_scales.append(m.noise_scale)

    running_fn = _compile_equation(config.cost.running, all_names,
                                     ordered_symbols, const_values,
                                     "Running cost", errors)
    terminal_fn = _compile_equation(config.cost.terminal, all_names,
                                      ordered_symbols, const_values,
                                      "Terminal cost", errors)

    if errors:
        raise PlantCompilationError(errors)

    nx, nu, n_p, ny = config.nx, config.nu, config.n_p, config.ny
    f_diag = [s.process_noise for s in config.states]
    fp_diag = [p.process_noise for p in config.parameters]
    state_clamps = {i: (s.min, s.max) for i, s in enumerate(config.states)
                     if s.min is not None or s.max is not None}

    continuous_dynamics_fn = make_continuous_dynamics_fn(
        nx, nu, n_p, xdot_fns, f_diag, fp_diag)
    dynamics_fn = make_dynamics_fn(continuous_dynamics_fn, state_clamps)
    measurement_fn = make_measurement_fn(nx, nu, n_p, ny, meas_fns, noise_scales)
    cost_fn = make_cost_fn(nx, nu, n_p, running_fn, terminal_fn)

    prior = np.array([p.prior_guess for p in config.parameters], dtype=float)
    constants = GenericConstants(prior=prior)

    return CompiledPlant(
        dynamics=dynamics_fn, measurement=measurement_fn, cost=cost_fn,
        continuous_dynamics=continuous_dynamics_fn, constants=constants,
        nx=nx, nu=nu, n_p=n_p, ny=ny)
