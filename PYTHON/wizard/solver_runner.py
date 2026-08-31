"""
Runs the wizard's "Run Simulation" step: compile the config
(generic_plant.py) then call the REAL, UNMODIFIED
extensions/dual_control/main_outer_control_loop() -- exactly what
applications/dynamic_pricing/run_pricing_demo.py and
applications/duopoly_pricing/run_duopoly_demo.py already do by hand.
Nothing about that function is changed or wrapped in a way that alters
its behavior; this module only adds the "never crash, never hang
silently" guarantees the wizard promises around calling it.

TIMEOUT DESIGN (documented limitation, not a claimed fix): the config's
own n_sessions/max_du_iterations are already hard-capped by schema.py
(MAX_N_SESSIONS/MAX_DU_ITERATIONS) -- THAT bound is the primary defense
against a runaway run. The wall-clock timeout here is a backstop on top
of that, implemented by running main_outer_control_loop() in a background
thread and giving up waiting after `timeout_seconds`. Because the call is
a long blocking sequence of plain Python/NumPy operations, Python cannot
forcibly kill the thread if the timeout fires -- the abandoned computation
may keep running in the background for a while even after this function
has already returned a timeout report to the caller. This is the same
"can't cleanly kill a blocking numpy call" limitation any pure-Python
timeout wrapper has; the input caps are what actually keeps worst-case
runtime bounded, not this timeout.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from extensions.dual_control.main_outer_control_loop import main_outer_control_loop

from wizard.generic_plant import compile_plant, PlantCompilationError
from wizard.schema import ModelConfig

DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass
class SolverReport:
    ok: bool
    result: Optional[dict] = None
    messages: list[str] = field(default_factory=list)
    raw_error: Optional[str] = None
    elapsed_seconds: float = 0.0


def _translate_solver_exception(e: Exception) -> str:
    if isinstance(e, np.linalg.LinAlgError):
        return ("The solver hit a numerical problem while tracking "
                "uncertainty (a covariance matrix was not valid). This can "
                "happen with noise scales set to zero, or with extreme "
                "bound/constant values -- try adjusting the process/"
                "measurement noise scale settings.")
    if isinstance(e, OverflowError):
        return ("A number in the calculation grew too large to handle. "
                "This usually means an equation (e.g. one using exp) can "
                "grow without bound for some allowed input values.")
    if isinstance(e, ZeroDivisionError):
        return "An equation divided by exactly zero during the run."
    if isinstance(e, ValueError):
        return f"The solver could not run with these settings: {e}"
    return (f"The solver could not complete ({type(e).__name__}). See "
            f"technical details below.")


def run_solver(config: ModelConfig,
                timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> SolverReport:
    """Never raises -- always returns a SolverReport."""
    try:
        plant = compile_plant(config)
    except PlantCompilationError as e:
        return SolverReport(ok=False, messages=list(e.errors))
    except Exception as e:  # noqa: BLE001
        return SolverReport(
            ok=False,
            messages=["Something unexpected went wrong while building "
                       "your model from its equations."],
            raw_error=f"{type(e).__name__}: {e}")

    x0 = np.array([s.initial_value for s in config.states], dtype=float)
    p_hat_0 = np.array([p.prior_guess for p in config.parameters], dtype=float)
    p_true = np.array(
        [p.true_value if p.true_value is not None else p.prior_guess
         for p in config.parameters], dtype=float)
    # cov_X fixed at 1.0 -- matches BOTH proven example demos exactly
    # (dynamic_pricing/duopoly_pricing hardcode this same value); states
    # themselves are never estimated by this layer, only unknown
    # parameters are, so there is no per-state prior-uncertainty concept
    # to expose here.
    cov_X = 1.0
    cov_P = (np.diag([p.prior_variance for p in config.parameters])
              if config.n_p > 0 else np.zeros((0, 0)))
    u_lims = np.array([[a.min, a.max] for a in config.actions], dtype=float)

    s = config.solver
    # augment_states_in_ilqg/filter are derived, never independently
    # configurable -- see schema.SolverSettings's docstring for the two
    # separate reasons True is required when n_p>0 and False is required
    # when n_p==0, given main_outer_control_loop.py's own (unmodified)
    # behavior in each mode.
    augment = config.n_p > 0
    solver_constants = plant.constants if augment else plant.constants.prior

    def _call():
        return main_outer_control_loop(
            T=config.n_sessions * config.dt, dt=config.dt,
            x_hat_0=x0, x_true_0=x0.copy(), p_hat_0=p_hat_0, p_true=p_true,
            cov_X=cov_X, cov_P=cov_P, constants=solver_constants,
            u_lims=u_lims, u_lim_method=s.u_lim_method,
            dynamics=plant.dynamics, measurement=plant.measurement,
            cost=plant.cost, continuous_dynamics=plant.continuous_dynamics,
            ny=plant.ny, nv=plant.ny, reg_type=s.reg_type,
            max_du_iterations=s.max_du_iterations,
            first_run_max_du_iterations=s.first_run_max_du_iterations,
            augment_states_in_ilqg=augment, augment_states_in_filter=augment,
            verbose=False)

    start = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        result = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)  # see module docstring: doesn't kill the thread
        elapsed = time.monotonic() - start
        return SolverReport(
            ok=False, elapsed_seconds=elapsed,
            messages=[f"This run took longer than {timeout_seconds:.0f} "
                       f"seconds and was stopped. Try lowering the number "
                       f"of steps to simulate, or the planning-iteration "
                       f"limits under Advanced settings."])
    except Exception as e:  # noqa: BLE001
        executor.shutdown(wait=False)
        elapsed = time.monotonic() - start
        return SolverReport(
            ok=False, elapsed_seconds=elapsed,
            messages=[_translate_solver_exception(e)],
            raw_error=f"{type(e).__name__}: {e}")
    else:
        executor.shutdown(wait=False)

    elapsed = time.monotonic() - start

    bad_keys = [k for k, v in result.items()
                 if isinstance(v, np.ndarray) and not np.isfinite(v).all()]
    if bad_keys:
        return SolverReport(
            ok=False, result=result, elapsed_seconds=elapsed,
            messages=[f"The solver produced invalid numbers (NaN/Inf) in: "
                       f"{', '.join(bad_keys)}. This usually means an "
                       f"equation divides by something that reaches zero, "
                       f"or a noise/bound setting is too extreme. Try "
                       f"adjusting the equations, bounds, or noise scales."])

    return SolverReport(ok=True, result=result, elapsed_seconds=elapsed)
