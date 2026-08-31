# Plug-In Guide: Bringing Your Own Model into This Framework

This guide is for someone who has never seen this codebase before and
wants to plug their own dynamical system ("plant") into the optimal
control / dual-control solver here, without needing to ask anyone for
help. It assumes you're comfortable with Python and NumPy, but assumes
**nothing** about this specific repository — including control-theory
vocabulary; unfamiliar terms are defined where they first matter.

If you only read three sections, read [§0](#0-one-time-setup) (or you'll
hit `ModuleNotFoundError` on the very first import), [§3](#3-the-exact-contract-what-your-plant-must-implement)
(the contract), and [§6](#6-validate-before-you-run-validate_pluginpy)
(the validator).

---

## Table of contents

0. [One-time setup](#0-one-time-setup)
1. [What this framework does](#1-what-this-framework-does)
2. [What "plugging in a model" means](#2-what-plugging-in-a-model-means)
3. [The exact contract: what your plant must implement](#3-the-exact-contract-what-your-plant-must-implement)
4. [Worked example: building `applications/dynamic_pricing/`](#4-worked-example-building-applicationsdynamic_pricing)
5. [Worked example: the plain core layer (`applications/todorov_toy/`)](#5-worked-example-the-plain-core-layer-applicationstodorov_toy)
6. [Validate before you run: `validate_plugin.py`](#6-validate-before-you-run-validate_pluginpy)
7. [Common mistakes, and how the validator catches them](#7-common-mistakes-and-how-the-validator-catches-them)
8. [Quick-reference checklist](#8-quick-reference-checklist)

---

## 0. One-time setup

This repository is organized as three real Python packages under
`PYTHON/`: `core/`, `extensions/`, `applications/` (see the project's
own `CLAUDE.md` "Repository layout" section for the full breakdown).
Cross-module imports are absolute package imports, e.g.
`from core.ddp_solver.ilqg import ilqg` — there is no `sys.path.append`
hack to make these resolve.

Before you write or run anything that imports from
`core`/`extensions`/`applications`, register the package once:

```
cd PYTHON
pip install -e .
```

This reads `PYTHON/pyproject.toml` (it lists `core*`, `extensions*`,
`applications*` as the packages to find) and installs the tree in
"editable" mode — so `import core.ddp_solver.ilqg` resolves from
anywhere, including a plain `python some_script.py` run from a
different working directory, with no need to re-run the install after
editing existing files (only needed again if you add a brand-new
top-level package directory). Skipping this step is why you'd hit
`ModuleNotFoundError: No module named 'core'` (or `extensions`/
`applications`) on the very first import in any of the worked examples
below.

If you're setting up a new plant, also create its own package
directory before writing code into it:

```
applications/my_plant/__init__.py   # empty file is fine
```

(an empty `__init__.py` is all that's required — see e.g.
`applications/dynamic_pricing/__init__.py`).

### A short glossary (terms used before they're formally defined)

A few terms recur throughout this guide before their full definition —
collected here so you don't have to hunt for a first-use explanation
mid-section:

- **Batch axis `K`** — every array the solver hands your functions
  carries a trailing `K` dimension because the solver evaluates many
  candidate trajectories in one vectorized call, not one at a time. Two
  concrete sources of a `K > 1` call: (a) **line search** — after each
  backward pass, `forward_pass.py` tries several candidate step sizes
  (`alphas`) simultaneously to see which one actually reduces cost, so
  `K` = number of step sizes tried that round; (b) **sigma points** —
  the SPKF (square-root sigma-point/unscented Kalman filter used by the
  dual-control layer's `todorov_estimator.py`/`spkf_function.py`)
  represents a probability distribution over the state/parameter
  estimate as a small deterministic set of sample points ("sigma
  points") instead of a single mean, and propagates all of them through
  your `dynamics`/`measurement` functions in one batched call, so `K` =
  number of sigma points that round. Your function must be vectorized
  over this axis — no Python loop over columns — and must not
  special-case `K == 1`.
- **MPC / receding horizon** — the dual-control layer doesn't solve one
  control problem once; `main_outer_control_loop()` re-solves a fresh
  iLQG problem at every session/step, executes only the first action
  from that solution, observes what actually happened, updates its
  parameter estimate, and repeats. "Session `k`'s plan" is a full
  trajectory, but only its first action is ever actually used.
- **Augmented state** — the dual-control layer's trick for online
  parameter estimation: instead of treating an unknown parameter as a
  fixed constant, it's appended as extra rows onto the state vector
  (`xa = [x; p]`) and given its own (typically slow, near-zero-drift)
  dynamics, so the same iLQG/SPKF machinery that estimates state also
  estimates the parameter. `augment_states=True` means `xa` includes
  those extra rows; `augment_states=False` means it doesn't, and your
  functions must handle both (see §3.1).
- **Dual control / curiosity** — with the state augmented this way, the
  controller's cost-to-go depends on the parameter *estimate's
  uncertainty*, not just its point value, so an action that's
  informative (shrinks that uncertainty fast) can be preferred over one
  that's merely optimal under the current point estimate. This is the
  literal mechanism behind this project's "curiosity-driven" framing
  (see the project's `CLAUDE.md` "Goal" section).

## 1. What this framework does

At its core, this repository is an **optimal control solver**: given a
system that evolves over time (a "plant") and a way of scoring how good
a trajectory is (a "cost"), it computes a sequence of actions that
drives the system toward low cost. The algorithm family is
**iLQG / DDP** (iterative Linear-Quadratic-Gaussian / Differential
Dynamic Programming) — it works by repeatedly linearizing the dynamics
and quadratizing the cost around a nominal trajectory, solving that
local approximation exactly, and shrinking the trust region (line
search) until it converges.

There are two layers you can plug a model into:

- **`core/ddp_solver/`** — the plain, deterministic version (Tassa,
  Mansard & Todorov's control-limited DDP/iLQG). You supply a
  simulation step and its derivatives; it hands back an optimal open-loop
  control sequence. No noise, no parameter estimation.
- **`extensions/dual_control/`** — builds on top of the same math but
  adds **stochastic dynamics** (process/measurement noise), **online
  parameter estimation** (a square-root sigma-point/unscented Kalman
  filter, SPKF), and **dual control**: the controller doesn't just act
  to minimize cost under its current best guess of unknown parameters —
  it also weighs how much a given action will *teach it about those
  parameters*. This is the "curiosity-driven" part of the project: an
  action that seems suboptimal under the current point estimate can
  still be chosen because it's informative. This layer re-plans on a
  receding horizon (MPC): at each session/step it solves a fresh iLQG
  problem, executes only the first action, observes what actually
  happened, updates its parameter estimate, and repeats.

**This guide is primarily about the `extensions/dual_control/` layer**,
because that's the one with online parameter estimation — the layer
most "plugging in a model" requests will actually want. §3 covers both
layers' contracts so you can pick the right one; §4 walks the
dual-control layer end to end, and §5 gives the plain layer an equally
complete, runnable worked example.

You never need to touch the solver internals (`ilqg_function.py`,
`backward_pass.py`, `forward_pass.py`, `todorov_estimator.py`,
`spkf_function.py`, `main_outer_control_loop.py`). They are
plant-agnostic — they take your dynamics/measurement/cost as plain
Python callables and never assume anything about what system you're
modeling. Plugging in a new model means writing those callables, not
modifying the solver.

## 2. What "plugging in a model" means

Concretely, plugging in a model means writing a small number of
**plain functions with a fixed calling convention** — how many
arguments, what shape each array is, what the function returns — so
that the solver can call them without knowing anything about your
specific system. This is enforced by convention (Python duck typing),
not by subclassing or registering anything: there is no base class to
inherit from and no plugin registry to add yourself to. You write
functions matching a documented signature, then hand those functions
directly to the solver's entry point (`ilqg()` or
`main_outer_control_loop()`) as arguments.

The formal description of these calling conventions lives in two files
you should read before writing any code:

- `core/ddp_solver/protocols.py` — the contract for the plain
  deterministic layer.
- `extensions/dual_control/protocols.py` — the contract for the
  stochastic/dual-control layer.

These are `typing.Protocol` classes. **They are documentation, not
enforcement** — nothing in the solver imports them or runtime-checks
against them. Their entire purpose is to be the one authoritative place
that states the calling convention, instead of you having to
reverse-engineer it by reading `forward_pass.py`'s internals. Read the
Protocol's docstring for the function you're about to write; it tells
you exactly what arguments arrive, in what shape, and what you must
return.

Once you've written your functions, there are two more pieces:

- A **constants object** — a small dataclass (or plain array) carrying
  every plant-specific numeric parameter your functions need (mass,
  demand elasticity, a lookup table, whatever). This gets threaded
  through as an opaque `constants` argument that the solver never
  inspects.
- A **validation call** — `validate_plugin.py` runs synthetic data
  through your functions and checks that they honor the documented
  shapes/conventions, catching a large class of mistakes in one clear
  error message before you ever hand them to the real solver. Both
  layers have one (§6).

## 3. The exact contract: what your plant must implement

### 3.1 Shared notation and conventions

These apply to every function in both layers below:

- `nx` = physical state dimension, `n_p` = number of estimated
  (augmented) parameters (dual-control layer only), `nxa` = `nx + n_p`
  when augmented else `nx`.
- `nu`/`m` = control/action dimension (`nu` in the dual-control layer,
  `m` in the core layer — same thing, different name).
- `nv` = raw measurement-noise channel count, `ny` = observation
  dimension (dual-control layer only).
- `K` = batch size — the solver evaluates many candidate trajectories
  in one call (e.g. every backtracking line-search step size at once,
  or every sigma point at once — see §0's glossary), so every array
  carries a trailing `K` axis. Your function must be vectorized over it
  — no Python loops over columns. **`K` is not fixed**: it can be `1`
  (a single trajectory) or several dozen depending on what the solver
  is doing internally at that moment. Don't write code that only works
  for a specific `K` (e.g. anything relying on `.squeeze()` to drop a
  size-1 batch axis) — the validator in §6 always tests with `K=3`, so
  a `K=1`-only bug would pass validation and fail later in real use.
- **Row ordering inside `xa`**: when a plant is augmented (dual-control
  layer, `augment_states=True`), the physical-state rows always come
  **first** (`xa[0:nx, :]`), and the estimated-parameter rows come
  **after** (`xa[nx:nx+n_p, :]`), in the same order your
  `continuous_dynamics` stacks them into `xdot`. This holds regardless
  of how many states or parameters you have — the worked example in §4
  only shows `nx=1, n_p=1`, but a plant with `nx=6, n_p=3` follows the
  identical rule: rows `0..5` are state, rows `6..8` are parameters, in
  whatever order your own `xdot = np.vstack([f, fp])`-style
  concatenation puts them.

### 3.2 The dual-control layer (`extensions/dual_control/protocols.py`)

You implement **four callables**. Each is a `typing.Protocol` class in
that file; what follows is a condensed version — read the file itself
for the full reasoning behind each convention.

#### `dynamics(dt, xa, u, constants, w, noise_index, augment_states, w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method) -> (nxa, K) ndarray`

The one-step-ahead **discrete** state transition.

| Argument | Shape / type | Meaning |
|---|---|---|
| `dt` | `float` | integration timestep |
| `xa` | `(nxa, K)` | current (possibly augmented) state batch |
| `u` | `(nu, K)` | control batch |
| `constants` | plant-defined | your constants object; never inspected by the solver |
| `w` | `(nw, K)` **or** `(nxa, K)` | see `w_from_filter` below — shape depends on it |
| `noise_index` | `False`, `0`, or 1-indexed `int i` | see below |
| `augment_states` | `bool` | whether `xa` includes the estimated-parameter rows |
| `w_from_filter` | `0`/`False` or `1`/`True` | see below |
| `dyn_noise_reg_lambda` | `float` | regularization scalar folded into noise scale; you may ignore it if your noise is not iteration-regularized (accept it anyway, for signature compatibility) |
| `u_lims` | `(nu, 2)` or `None` | control bounds |
| `u_lim_method` | `1` or `2` | `1`: bounds enforced externally by box-QP — your function ignores `u_lims` entirely. `2`: bounds enforced internally — you must `tanh`-squash `u` into `u_lims` yourself before using it |

`noise_index` gotcha: `False` **and literal `0`** both mean "no
specific noise column requested, give me the actual next state."
Different call sites in this codebase pass one or the other
inconsistently — you **must** branch with `if noise_index:` (truthy),
never `if noise_index is False:`, or you will silently mishandle half
the call sites. A 1-indexed `int i` means "don't step the state at all
— instead return the `i`-th column of the dynamics noise-diffusion
matrix, broadcast across all `K` columns" (this is what the solver
differentiates to build noise-covariance Jacobians).

`w_from_filter` gotcha: when `0`/`False`, `w` is a **raw noise sample**
of shape `(nw, K)` that you must scale yourself by your own diffusion
matrix and `sqrt(dt)` (Euler–Maruyama convention). When `1`/`True`, `w`
**is already** the final additive perturbation, shape `(nxa, K)` — add
it directly, no further scaling. These two shapes are genuinely
different (`nw` need not equal `nxa` in general, though they coincide
in the reference examples) — your function must handle both.

Return: `(nxa, K)` ndarray — next state (ordinary call) or a
noise-diffusion column tiled across `K` (`noise_index` request).

#### `measurement(dt, xa, u, constants, v, noise_index, augment_states, v_from_filter) -> (ny, K) ndarray`

The observation model. Same `noise_index`/`v_from_filter` conventions
as `dynamics`'s `w`/`w_from_filter`, but for the measurement side —
`v` is `(nv, K)` when `v_from_filter` is falsy, `(ny, K)` when truthy.
`u` is accepted for signature compatibility even if your observation
model doesn't depend on the control (it's fine to leave it unused).
`nv` and `ny` need not be equal in general (they coincide in every
reference example here, which is why that distinction is easy to
miss) — if your plant observes fewer/more channels than it has raw
noise sources, keep `nv` and `ny` as the separate quantities they are.

Return: `(ny, K)` ndarray.

#### `cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants) -> (K,) ndarray`

Running/terminal cost, differentiated (via finite differences) by the
solver to build its local quadratic approximation.

| Argument | Shape / type | Meaning |
|---|---|---|
| `xa` | `(nxa, K)` | state batch |
| `u` | `(nu, K)` | control batch — **columns may be all-`NaN`** to signal "this is the terminal step, score final cost only." |
| `tracking_trajectory` | `Any`, commonly `None` | reserved for a time-varying reference; ignore if your cost doesn't track one |
| `u_lims`, `u_lim_method` | as above | if `u_lim_method == 2`, you must apply the **same** tanh-squash internally that your `dynamics` does, so cost and dynamics agree on what "the control" physically means |
| `constants` | plant-defined | |

**You must detect the all-`NaN` terminal columns and zero them out
before using `u`** — it is not safe to assume `u` is finite. This is
called out explicitly in the Protocol docstring as *the single most
common plug-in bug*, and the validator has a dedicated check for it
(§6).

Return: `(K,)` ndarray, no `NaN`s.

#### `continuous_dynamics(dt, xa, u, constants, augment_states, u_lims, u_lim_method) -> (xdot, nx, n_p, F, Fp)`

The continuous-time vector field `dx/dt = f(x, u)`, used once per outer
MPC iteration purely to build the parameter-noise-scale matrices fed
into the estimator. This is **not** the same as `dynamics` — it returns
a derivative, not a next state.

Return, a 5-tuple:

| Element | Shape | Meaning |
|---|---|---|
| `xdot` | `(nx [+ n_p], K)` | state (and parameter, if augmented) derivatives |
| `nx` | `int` | physical state dimension, constant across calls |
| `n_p` | `int` | number of augmented parameters; `0` if `augment_states` is `False` |
| `F` | `(nx, nx)` | dynamics noise-scale matrix for the physical states |
| `Fp` | `(n_p, n_p)` | parameter noise-scale matrix; shape `(0, 0)` if not augmented |

A common (not required) pattern: implement `continuous_dynamics` first,
then have `dynamics` call it and Euler-integrate the result
(`xa + dt*xdot + noise`) — this is what the worked example in §4 does.

### 3.3 The plain deterministic layer (`core/ddp_solver/protocols.py`)

Use this instead if you don't need noise or online parameter
estimation — just "solve for the best open-loop action sequence." Two
required callables, one optional:

- **`step_fn(x, u, i) -> (x_next, cost)`** — `x` is `(n, K)`, `u` is
  `(m, K)`. At the final step (`i == N`, one past the last control
  step) `u` arrives as all-`NaN`; detect and zero it, same convention
  as `cost` above. Returns `(n, K)` next state and `(K,)` stage cost.
- **`derivs_fn(x, u) -> (fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu)`**
  — `x`/`u` are the *full* nominal trajectory, shape `(n, N+1)`/
  `(m, N+1)` (note: `N` here means "number of control steps," so the
  trajectory has `N+1` points — see the file's own naming-gotcha note).
  `fxx`/`fxu`/`fuu` must be either **all** `None` (iLQG mode) or **all**
  provided (full DDP, second-order dynamics) — never a mix.

  > **`N` means something different in each layer.** In this core layer,
  > `N` = number of control steps (a trajectory has `N+1` points). In
  > §4's dual-control layer, `N = round(T/dt)` counts **sessions**
  > instead, and the two don't line up the same way (`u` there has
  > `N-1` columns, not `N+1`). They're unrelated quantities that happen
  > to share a letter — don't carry an assumption from one layer's `N`
  > into the other's.
- **`constraint_fn(x_traj, u_traj) -> lims`** (optional) — per-timestep
  control bounds, shape `(N, m, 2)`, for the Dastan & Sensinger
  constrained-DDP extension. Only needed if your problem has
  state-dependent control constraints.

The solver entry point for this layer is `core.ddp_solver.ilqg.ilqg`;
see §5 for a complete, runnable call.

## 4. Worked example: building `applications/dynamic_pricing/`

This walks through how `applications/dynamic_pricing/` was actually
built, end to end, as a template for your own plant. The problem: a
seller sets a price each period; demand relaxes toward a price-implied
level with an **unknown** price-sensitivity parameter `b`, which the
dual-control loop estimates online while pricing near-optimally.

State: `nx = 1` (realized demand `x1`). Action: `nu = 1` (price).
Augmented parameter: `n_p = 1` (`b_hat`, price sensitivity).
Observation: `ny = 1` (demand is observed directly, with small noise).

Before Step 1, make sure §0 is done: `pip install -e .` has been run,
and `applications/dynamic_pricing/__init__.py` (or, for your own plant,
`applications/my_plant/__init__.py`) exists.

#### Step 1 — Define the constants object

Every plant needs a small object carrying its numeric parameters,
threaded through as the `constants` argument every one of the four
callables accepts. `applications/dynamic_pricing/pricing_constants.py`:

```python
from dataclasses import dataclass

@dataclass
class PricingConstants:
    a: float          # demand intercept at price 0 (known)
    c: float          # marginal cost per unit (known)
    eta: float        # demand partial-adjustment rate, in (0, 1] (known)
    p_ref: float      # reference price for the cost regularizer
    b_prior: float    # nominal/prior price sensitivity (used when augment_states=False)
    F_dyn: float = 1.0       # dynamics noise scale on x
    F_param: float = 0.05    # parameter (b) noise scale -- random-walk diffusion
    sigma_y: float = 1.0     # measurement noise scale on observed demand
    lambda_reg: float = 1e-4  # cost regularizer weight (QP conditioning only)
```

This is a plain dataclass — the solver never looks inside it. Only
your own dynamics/measurement/cost functions read its fields. There is
no formula for picking `F_dyn`/`F_param`/`sigma_y`/`lambda_reg` — these
are tuning knobs, not part of the plug-in contract. One rule of thumb
used here: keep `F_param` small relative to `F_dyn`, so the estimated
parameter drifts slowly session-to-session instead of as fast as the
physical state itself.

#### Step 2 — Write `continuous_dynamics.py`

Start with the continuous-time physics: demand relaxes toward the
price-implied demand curve `D(u;b) = a - b*u` at rate `eta`; the
unknown parameter `b` is a random walk (zero drift) while being
estimated.

```python
import numpy as np

def continuous_dynamics(dt, xa, u, constants, augment_states, u_lims, u_lim_method):
    nx = 1
    K = xa.shape[1]
    a, eta = constants.a, constants.eta
    x = xa[0, :]
    price = u[0, :]

    if augment_states:
        b = xa[1, :]
        n_p = 1
    else:
        b = np.full(K, constants.b_prior)
        n_p = 0

    f = (eta * (a - b * price - x)).reshape(1, K)
    xdot = np.vstack([f, np.zeros((n_p, K))]) if augment_states else f

    F = np.array([[constants.F_dyn]])
    Fp = np.array([[constants.F_param]]) if augment_states else np.zeros((0, 0))
    return xdot, nx, n_p, F, Fp
```

*(Condensed — the real file also broadcasts a size-1 `u` across `K`
columns and applies the `u_lim_method == 2` tanh-squash before using
`price`; both are omitted here since this demo only ever runs with
`u_lim_method=1`. If you use `u_lim_method=2`, you need that branch —
see `applications/dynamic_pricing/continuous_dynamics.py` for the full
version.)*

Note the pattern used throughout: `augment_states` decides whether `b`
is read from the state (`xa[1, :]`) or from the constants' fixed prior
(`constants.b_prior`) — the same function must handle both, since the
solver calls it both ways at different points in the pipeline.

#### Step 3 — Write `dynamics.py`, wrapping step 2

`continuous_dynamics` gives a derivative, not a next state. `dynamics.py`
Euler-integrates it and adds noise, matching `DynamicsFn`'s exact
signature:

```python
import numpy as np
from applications.dynamic_pricing.continuous_dynamics import continuous_dynamics

def dynamics(dt, xa, u, constants, w, noise_index, augment_states,
             w_from_filter, dyn_noise_reg_lambda, u_lims, u_lim_method):
    fa, nx, n_p, F, Fp = continuous_dynamics(
        dt, xa, u, constants, augment_states, u_lims, u_lim_method)
    F_a = _block_diag(F, Fp)  # (nxa, nxa)

    if noise_index:  # truthy check -- NOT `is False`
        col = F_a[:, noise_index - 1]
        return np.tile(col[:, None], (1, xa.shape[1]))

    noise = (F_a @ w) * np.sqrt(dt) if not w_from_filter else w
    x_next = xa + fa * dt + noise
    x_next[0, :] = np.maximum(x_next[0, :], 0.0)  # demand can't go negative
    return x_next
```

*(Condensed — the real file also broadcasts `w` and `u` across `K`
columns when either arrives with a single column
(`if w.shape[1] == 1 and Kplus1 > 1: w = np.tile(w, (1, Kplus1))`, same
for `u`), so a caller that passes an unbatched noise sample or control
still works against a batched `xa`. Omitted here for the same reason as
Step 2's — see `applications/dynamic_pricing/dynamics.py` for the full
version.)*

Two contract details worth calling out because they're easy to get
wrong the first time:
- `if noise_index:` — a truthy check, so it correctly treats both
  `False` and `0` as "no column requested" (see §3.2's gotcha).
- The floor clamp (`x >= 0`) is applied to the **discrete** next state
  here, not inside `continuous_dynamics`'s derivative — the continuous
  vector field is left unclamped so it stays smooth for the estimator's
  Jacobians; the clamp is a property of the discrete step only.

(`_block_diag` is a small local helper that builds a block-diagonal
matrix from `F`/`Fp` — see the real file for its few lines; nothing
about it is part of the contract.)

#### Step 4 — Write `measurement.py`

The observation model — here, demand is observed directly with small
noise:

```python
import numpy as np

def measurement(dt, xa, u, constants, v, noise_index, augment_states, v_from_filter):
    G = np.array([[constants.sigma_y]])  # (ny, nv) = (1, 1)
    if noise_index:
        return np.tile(G[:, noise_index - 1][:, None], (1, xa.shape[1]))
    g = xa[0:1, :]  # observe realized demand x1 directly
    noise = (G @ v) * np.sqrt(dt) if not v_from_filter else v
    return g + noise
```

`u` and most of `constants` are unused here — that's fine. Accept them
anyway for signature compatibility; every call site passes them
regardless of whether your observation model needs them.

#### Step 5 — Write `cost.py` — and the real mistake that was made here

```python
import numpy as np

def cost(xa, u, tracking_trajectory, u_lims, u_lim_method, constants):
    u = np.array(u, dtype=float, copy=True)
    final = np.isnan(u[0, :])
    u[:, final] = 0.0                      # <-- required: zero the NaN terminal marker

    price = u[0, :]
    K = xa.shape[1]
    b_hat = xa[1, :] if xa.shape[0] > 1 else np.full(K, constants.b_prior)

    demand_at_price = constants.a - b_hat * price   # CURRENT price-implied demand
    profit = (price - constants.c) * demand_at_price
    reg = constants.lambda_reg * (price - constants.p_ref) ** 2

    total = -profit + reg
    total[final] = 0.0
    return total
```

*(Condensed — the real file also applies the `u_lim_method == 2`
tanh-squash to `u` before computing `price`, mirroring
`continuous_dynamics.py`'s branch; omitted here for the same reason.)*

The first version of this file priced profit off the **lagged** state
(`x1`, this period's already-realized demand) instead of the
**current** price-implied demand `D(u; b_hat)`. That version passed
every shape/signature check the validator runs — it is a perfectly
valid `CostFn` — but it was economically broken: under this framework's
receding-horizon MPC (only the first action of each freshly re-planned
trajectory is ever executed), pricing off the lagged state gives the
**current** action zero within-period cost for being high, since it
can't affect demand that already happened. The controller pinned price
at the ceiling every single session, and no amount of regularizer
tuning fixed it (even a 100,000x larger `lambda_reg` only nudged price
from the ceiling to ~$35 instead of the true optimum ~$30). The fix was
structural — price off the *current* demand curve instead — not a
tuning problem.

**Lesson for your own plugin: the validator checks contract compliance,
not whether your cost function actually incentivizes the right thing.**
A shape-correct, NaN-safe cost function can still be economically or
physically nonsensical. Passing validation is necessary, not
sufficient — you still need to sanity-check the plant's actual
behavior (Step 7 below).

#### Step 6 — Validate your plug-in: `test_plugin_contract.py`

Before wiring these four functions into the real solver, run them
through the dual-control layer's validator — it catches shape/
signature/NaN-hygiene mistakes (including the `noise_index`/
`w_from_filter` gotchas from §3.2) in one pass, with one clear error
listing everything wrong at once:

```python
from applications.dynamic_pricing.continuous_dynamics import continuous_dynamics
from applications.dynamic_pricing.cost import cost
from applications.dynamic_pricing.dynamics import dynamics
from applications.dynamic_pricing.measurement import measurement
from applications.dynamic_pricing.pricing_constants import PricingConstants
from extensions.dual_control.validate_plugin import validate_dual_control_plugin


def make_constants():
    return PricingConstants(
        a=100.0, c=10.0, eta=0.3, p_ref=35.0, b_prior=2.0,
        F_dyn=1.0, F_param=0.05, sigma_y=1.0, lambda_reg=1e-4)


def test_dynamic_pricing_plugin_satisfies_dual_control_protocols():
    # nx=1 (realized demand), nu=1 (price), n_p=1 (price sensitivity b),
    # nv=1/ny=1 (demand observed directly, one noise channel)
    validate_dual_control_plugin(
        dynamics, measurement, cost, continuous_dynamics,
        nx=1, nu=1, nv=1, ny=1, constants=make_constants(), n_p=1)
```

This is a real `pytest` test
(`applications/dynamic_pricing/test_plugin_contract.py`), not a
one-off script — it re-runs automatically alongside the rest of the
suite, so a later edit to any of the four functions that breaks the
contract fails immediately instead of surfacing three layers deep
inside `forward_pass.py`. Run it directly with:

```
pytest applications/dynamic_pricing/test_plugin_contract.py -v
```

`pytest`'s normal `1 passed` output means every check succeeded — the
validator itself prints nothing on success (§6). If you don't want to
use pytest, the same check can be run as a one-off script:

```
python -c "from applications.dynamic_pricing.test_plugin_contract import test_dynamic_pricing_plugin_satisfies_dual_control_protocols as t; t()"
```

On failure the test fails with a `PluginContractError` whose message
lists every issue found, not just the first. See §6 for exactly what
`validate_dual_control_plugin` checks, and §7 for what each specific
failure message means.

#### Step 7 — Run it for real: `main_outer_control_loop`

Validation passing only proves contract compliance (§6) — it says
nothing about whether the plant's behavior is actually sensible. The
last step is to wire the validated callables into the real solver entry
point and read the actual numbers it produces.

One conceptual point first: **`main_outer_control_loop` is a
closed-loop *simulation* harness, not a live-deployment driver.** Per
session it (1) re-plans via iLQG from the current *estimated* state,
(2) advances a second, separate "true" trajectory by calling your
`dynamics`/`measurement` functions again with the *true* parameter
value, simulating what would actually happen, and (3) fuses that
simulated observation back into the estimate via the SPKF (see the
per-iteration breakdown in `extensions/dual_control/main_outer_control_loop.py`'s
own module docstring). This is what lets you test "does the estimate
converge to the truth" without any real hardware or data — but it's why
the function needs *both* a believed and a true initial condition. If
you eventually connect this to a real system, you'd replace step (2)
with real sensor readings and drop `x_true_0`/`p_true` entirely —
that's a different driver than this one, and isn't covered by this
guide.

Argument-by-argument, matched to what you already built:

| Argument | Shape / type | Meaning | Value used here |
|---|---|---|---|
| `T` | `float` | total horizon = `n_sessions * dt` | `15 * 1.0` |
| `dt` | `float` | same `dt` your `dynamics`/`continuous_dynamics` expect | `1.0` |
| `x_hat_0` | `(nx,)` | the controller's initial **belief** about the physical state | the true starting demand — directly observable even though `b` isn't |
| `x_true_0` | `(nx,)` | the **actual** initial physical state, used to simulate truth | same value as `x_hat_0` — nothing hides the starting demand |
| `p_hat_0` | `(n_p,)` | the controller's initial **belief** about the unknown parameter(s) | `[0.5]` — deliberately wrong, to demonstrate convergence |
| `p_true` | `(n_p,)` | the **actual** parameter value used to simulate truth; the controller never sees this directly, only through observations | `[2.0]` |
| `cov_X` | scalar | how uncertain the initial state estimate is. The source builds the filter's covariance as `cov_X * np.eye(nx)` — an **elementwise** product — so pass a scalar; a full `(nx, nx)` matrix here does *not* give you a full covariance, it silently keeps only that matrix's diagonal | `1.0` |
| `cov_P` | `(n_p, n_p)` | how uncertain the initial parameter estimate is — should be wide enough that it doesn't rule out the true value given how wrong `p_hat_0` might be | `[[1.0]]` |
| `constants` | plant-defined | the object from Step 1 | — |
| `u_lims` | `(nu, 2)` | control bounds | `[[10.0, 45.0]]` |
| `u_lim_method` | `1` or `2` | see §3.2 — default to `1` (see the saturation pitfall below) | `1` |
| `dynamics, measurement, cost, continuous_dynamics` | callables | your four functions from Steps 2-5 | — |
| `ny, nv` | `int` | observation / raw-noise-channel dimensions | `1, 1` |
| `augment_states_in_ilqg` | `bool` | whether the planner treats the parameter as part of the state it optimizes over | `True` |
| `augment_states_in_filter` | `bool` | whether the estimator treats the parameter as part of the state it estimates | `True` |

For your own plant: `x_hat_0`/`x_true_0` are usually identical (the
physical state is directly observable at the start; only the parameter
is hidden). `p_hat_0` is your prior guess — make it deliberately wrong
if you want to see the estimate correct itself. `p_true` only exists
because this driver simulates the true system itself (see above) — set
it to whatever value you want to test convergence against.

`main_outer_control_loop` also takes further optional tuning parameters
beyond the required ones above. Its own docstring only documents the
return value, not these — so here's what each one actually does, read
directly from the function body in
`extensions/dual_control/main_outer_control_loop.py` rather than from
its docstring:

| Parameter | Default | What it actually does |
|---|---|---|
| `reg_type` | `3` | Which Levenberg–Marquardt scheme the inner iLQG backward pass uses to regularize the control-law solve (see `extensions/dual_control/backward_pass.py`): `1` adds `lam` directly to the control Hessian; `2` adds it to the value-function Hessian before it propagates into the control Hessian; `3` (default) shifts the control Hessian by its own most-negative eigenvalue, guaranteeing it's solvable; `4` floors every eigenvalue of the control Hessian at `lam`. |
| `lambda_i`, `dlambda_i` | `1.0`, `1.0` | The regularization strength `lam` used above, and its multiplicative step size — both re-used by every session's inner iLQG solve. |
| `reset_lambda` | `True` | Whether `lam`/`dlambda` are reset to `lambda_i`/`dlambda_i` before each session's replan, rather than carrying over whatever the previous session's solve ended on. |
| `max_du_iterations`, `first_run_max_du_iterations` | `250`, `500` | Cap on inner iLQG iterations per replan. The very first session gets a bigger budget by default, since it has no warm-started guess to start from. |
| `dyn_noise_reg` | `0.0` | Multiplied by `lam` and passed straight through as `dynamics`'s `dyn_noise_reg_lambda` argument (§3.2). |
| `keep_dyn_noise`, `keep_meas_noise` | `False`, `False` | Whether the internally *simulated "true" system* (step (2) above) actually receives dynamics/measurement noise when it's stepped, or advances deterministically despite being the "true" trajectory. |
| `keep_dyn_noise_filter`, `keep_meas_noise_filter` | `True`, `True` | Same idea, but for whether the *estimator's* covariance propagation (the SPKF re-estimation in step (3)) accounts for dynamics/measurement noise. |
| `horizon_mode` | `2` | `1` = shrinking horizon (the re-plan window gets shorter each session as `T_rem` counts down). `2` = rolling/receding horizon (every session re-plans over the same fixed-length window `T`, shifting the warm start forward by one step). |
| `h` | `sqrt(3)` | The SPKF's sigma-point spread parameter (§0's glossary). |
| `additive_noise` | `False` | Passed straight through to the SPKF, selecting additive vs. non-additive noise handling. |
| `first_run_seeds` | `1` | Must stay `1` — a multi-seed search over the first session's solution isn't implemented in this port; anything else raises `NotImplementedError`. |
| `nw` | `None` | Dynamics noise-channel count fed to `dynamics`'s `w` argument (§3.2). Left `None`, it's computed automatically as `nx+n_p` (if `augment_states_in_ilqg`) or `nx`. |
| `u_bar_0` | `None` | Optional `(nu, N)` initial control guess for the **first** iLQG call only — every later call warm-starts from the previous solution's shifted trajectory regardless of this. Not in the original MATLAB source; added to let you nudge a non-convex cost landscape's first solve away from an all-zero local optimum that a bilinear (cross-term) cost surface can otherwise settle into. Left `None`, the first call warm-starts from all-zeros, same as before this parameter existed. |
| `verbose` | `False` | Print per-session progress. |

None of these need to change from their defaults to get the dynamic-pricing
demo working — they're listed here so you know what to reach for if your
own plant needs different behavior (e.g. a plant where the "true"
simulated system should be noiseless: set `keep_dyn_noise=False`, which
is already the default).

```python
from extensions.dual_control.main_outer_control_loop import main_outer_control_loop
from applications.dynamic_pricing.continuous_dynamics import continuous_dynamics
from applications.dynamic_pricing.cost import cost
from applications.dynamic_pricing.dynamics import dynamics
from applications.dynamic_pricing.measurement import measurement
from applications.dynamic_pricing.pricing_constants import PricingConstants

constants = PricingConstants(a=100.0, c=10.0, eta=0.3, p_ref=35.0, b_prior=0.5,
                              F_dyn=1.0, F_param=0.05, sigma_y=1.0, lambda_reg=1e-4)

result = main_outer_control_loop(
    T=15.0, dt=1.0,
    x_hat_0=x0, x_true_0=x0.copy(),            # x0: true starting demand
    p_hat_0=np.array([0.5]),                   # deliberately WRONG prior on b
    p_true=np.array([2.0]),                    # true b, unknown to the controller
    cov_X=1.0, cov_P=np.array([[1.0]]),
    constants=constants, u_lims=np.array([[10.0, 45.0]]), u_lim_method=1,
    dynamics=dynamics, measurement=measurement, cost=cost,
    continuous_dynamics=continuous_dynamics, ny=1, nv=1,
    reg_type=1, max_du_iterations=60, first_run_max_du_iterations=100,
    augment_states_in_ilqg=True, augment_states_in_filter=True)
```

`reg_type=1, max_du_iterations=60, first_run_max_du_iterations=100`
above are **not** the function's defaults (`3`/`250`/`500`) — the real
`run_pricing_demo.py` sets them explicitly, and this is not a
simplification: dropping back to the defaults changes the convergence
behavior described below.

**What you get back** — `main_outer_control_loop` returns a `dict`. Every
shape below uses `N = round(T/dt)` — the number of **sessions**, a
completely different quantity from the core layer's `N` (§3.3, §5:
"number of control steps," trajectory has `N+1` points). Don't carry
one layer's `N` into the other.

| Key | Shape | Meaning |
|---|---|---|
| `x_hat` | `(nx, N)` | estimated physical state at each session |
| `p_hat` | `(n_p, N)` | estimated parameter at each session |
| `x_true` | `(nx, N)` | simulated true physical state at each session |
| `xa_true` | `(nx+n_p or nx, N)` | simulated true augmented state |
| `u` | `(nu, N-1)` | the action actually applied each session — one fewer than `N`, since no action is taken at the final, terminal session |
| `y_true` | `(ny, N-1)` | the simulated observation at each session |
| `cost_true` | `(N,)` | true cost incurred per session |
| `cost_est` | `(N,)` | cost as evaluated at the filtered estimate (not necessarily equal to `cost_true`) |
| `total_true_cost` | `float` | `cost_true.sum()` |

Run directly with `python run_pricing_demo.py`
(`applications/dynamic_pricing/run_pricing_demo.py`). What to actually
check, using this plant as the template:

1. **Does the parameter estimate converge toward the truth?** Start
   `p_hat_0` deliberately wrong (here, `b_hat_0=0.5` vs. the true
   `b=2.0`) and confirm `abs(p_hat[:, -1] - p_true) <
   abs(p_hat[:, 0] - p_true)` — the estimate should end up closer than
   it started, not just "moved." This is the same qualitative proof
   already used to validate the underlying dual-control solver on
   Andrew's SIDARTHE plant (see `CLAUDE.md`'s Andrew STATUS entry) —
   this demo is the pricing plant's version of it.
2. **Are the chosen actions economically/physically sensible?** — not
   pinned at a bound every single session, staying inside `u_lims` with
   visible session-to-session variation as the estimate updates.

This exact demo caught two real bugs that `validate_dual_control_plugin`
(§6) had already passed cleanly on both broken versions, because both
were shape-correct and NaN-safe — see Step 5's lesson above, and each
bug's own diagnosis in `run_pricing_demo.py`'s module docstring and
`CLAUDE.md`'s dynamic-pricing review:

- **Pricing off the wrong state.** The first `cost.py` priced profit
  off the *lagged* demand `x1` (this period's already-realized demand)
  instead of the *current* price-implied demand `D(u; b_hat)`. Under
  this framework's receding-horizon MPC (only the first action of each
  freshly re-planned trajectory is ever executed), pricing off a lagged
  state gives the current action *zero* within-period cost for being
  high — so the controller pinned price at the ceiling every session,
  and no amount of regularizer tuning fixed it. The fix was structural
  (price off the current demand curve), not a tuning problem — this is
  §4's own worked "real mistake," reproduced here as the case that
  motivated writing this step at all.
- **A `u_lim_method=2` saturation trap.** Even after fixing the cost,
  price stayed pinned at the ceiling. Root cause: `u_lim_method=2`
  enforces control bounds by `tanh`-squashing an unconstrained raw
  control into `u_lims`; a badly wrong initial belief (session 0)
  pushed the *raw pre-tanh* control deep into `tanh`'s saturated
  region, where `d(price)/d(u_raw) ≈ 0` — so gradient-based
  re-optimization in later sessions couldn't pull it back even after
  `b_hat` corrected itself (warm-starting carries the saturated raw
  value forward). Switching to `u_lim_method=1` (box-QP bound
  enforcement in `back_pass.py`, operating on price directly, no
  reparameterization) has no such saturation region, and price
  correctly snaps to the true monopoly-optimal ~$30 from session 1
  onward. **Takeaway for your own plugin:** if a controller stays
  pinned at a bound no matter how you tune the cost, suspect
  `u_lim_method=2`'s tanh reparameterization before suspecting the cost
  function itself — try `u_lim_method=1` first.

## 5. Worked example: the plain core layer (`applications/todorov_toy/`)

This is the equally-complete counterpart to §4 for the layer without
noise or online parameter estimation — `core/ddp_solver/`. Use this
layer when you just want the best open-loop action sequence for a
known, deterministic system: no `constants` object with a random-walk
parameter, no augmented state, no measurement model, no dual control.
Built from `applications/todorov_toy/run_todorov_demo.py`, which
reproduces Todorov's own `demo_linear.m`: a random marginally-stable
linear system with quadratic cost.

Two callables only (§3.3): `step_fn` and `derivs_fn`.

#### Step 1 — Define the system (no constants dataclass required)

Unlike §4's dual-control layer, this layer doesn't mandate a
`constants` argument at all — `step_fn`/`derivs_fn` take only
`(x, u, i)`/`(x, u)`. If your plant has fixed numeric parameters, the
simplest approach is exactly what this demo does: close over them from
the enclosing scope instead of threading them through as an argument.

```python
import numpy as np
from scipy.linalg import expm

def make_system(rng, n, m, h):
    A = rng.standard_normal((n, n))
    A = A - A.T                 # skew-symmetric -> purely imaginary eigenvalues
    A = expm(h * A)             # discrete time, marginally stable
    B = h * rng.standard_normal((n, m))
    Q = h * np.eye(n)
    R = 0.1 * h * np.eye(m)
    return A, B, Q, R
```

`n` = state dim, `m` = control dim, `h` = timestep. `A`/`B`/`Q`/`R` are
captured by the closures in Step 2 rather than passed explicitly on
every call.

#### Step 2 — Write `step_fn(x, u, i) -> (x_next, cost)`

The one-step **discrete** transition plus running/terminal cost in one
function (§3.3: this layer merges what the dual-control layer splits
across `dynamics`/`cost`). `x` is `(n, K)`, `u` is `(m, K)`; at the
terminal step (`i == N`) `u` arrives all-`NaN` — same convention, and
same "zero it before using it" requirement, as §3.2's `cost`:

```python
def make_step_and_derivs(A, B, Q, R):
    def step_fn(x, u, i):
        u = np.where(np.isnan(u), 0.0, u)      # required: zero the NaN terminal marker
        x_next = A @ x + B @ u
        cost = 0.5 * np.sum(x * (Q @ x), axis=0) + 0.5 * np.sum(u * (R @ u), axis=0)
        return x_next, cost
    ...
```

Both `x_next` `(n, K)` and `cost` `(K,)` are vectorized over the batch
axis `K` automatically here because `A @ x`/matrix broadcasting already
operate column-wise — no explicit loop needed for a linear system. A
nonlinear `step_fn` must be vectorized the same way by construction
(elementwise NumPy ops over the trailing axis), not with a Python loop
over columns (§0's glossary: `K` is not fixed and the validator always
tests `K=3`).

#### Step 3 — Write `derivs_fn(x, u) -> (fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu)`

Unlike `step_fn`, this one takes the **full nominal trajectory** at
once: `x` is `(n, N+1)`, `u` is `(m, N+1)` with the last column all-
`NaN` (§3.3's naming gotcha: `N` = number of control steps here, so the
trajectory has `N+1` points). For a linear-quadratic system every one
of these derivatives is a constant matrix, just repeated across the
trajectory's time axis:

```python
    def derivs_fn(x, u_aug):
        n_time = x.shape[1]
        u0 = np.where(np.isnan(u_aug), 0.0, u_aug)
        fx = np.repeat(A[:, :, None], n_time, axis=2)
        fu = np.repeat(B[:, :, None], n_time, axis=2)
        cx = Q @ x
        cu = R @ u0
        cxx = np.repeat(Q[:, :, None], n_time, axis=2)
        cxu = np.zeros((A.shape[0], B.shape[1], n_time))
        cuu = np.repeat(R[:, :, None], n_time, axis=2)
        return fx, fu, None, None, None, cx, cu, cxx, cxu, cuu

    return step_fn, derivs_fn
```

`fxx`/`fxu`/`fuu` are all `None` here — this system is linear, so its
dynamics have zero curvature, and iLQG (not full DDP) is the correct
mode. Per §3.3, these three must be **either all `None` or all
provided** — never a mix; `validate_derivs_fn` (§6) rejects a partial
set explicitly.

#### Step 4 — Call the solver directly: `core.ddp_solver.ilqg.ilqg`

No `main_outer_control_loop()`/MPC wrapper at this layer — you call
`ilqg()` once and get back an optimized open-loop trajectory:

```python
from core.ddp_solver.ilqg import ilqg

rng = np.random.default_rng(0)
n, m, N, h = 3, 2, 20, 0.05
A, B, Q, R = make_system(rng, n, m, h)
step_fn, derivs_fn = make_step_and_derivs(A, B, Q, R)

x0 = rng.standard_normal(n)
u0 = 0.1 * rng.standard_normal((m, N))   # initial guess, N control steps

x, u, L, Vx, Vxx, cost, trace, stop_reason = ilqg(
    step_fn, derivs_fn, x0, u0, lims=None, max_iter=50,
    tol_fun=1e-12, tol_grad=1e-10, verbose=1)
```

`x0` is `(n,)` (a single initial state, not batched — `ilqg()` builds
the batch internally for line search), `u0` is `(m, N)` (your initial
guess for the control sequence, `N` steps). `lims=None` means
unconstrained; pass `(m, 2)` bounds instead if your controls are
physically limited, or a `constraint_fn` for the state-dependent case
(§3.3, the Dastan & Sensinger extension — see
`PYTHON/extensions/constraints/`).

#### Step 5 — Sanity-check: cost should actually go down

`run_todorov_demo.py` (`applications/todorov_toy/run_todorov_demo.py`)
computes the initial trajectory's cost via `forward_pass()` directly
(the same function `ilqg()` calls internally for line search) before
optimizing, so you can see the improvement:

```python
from core.ddp_solver.forward_pass import forward_pass

x_init, _, cost_init = forward_pass(
    x0, u0, None, None, None, np.array([1.0]), step_fn, None)
cost_init_total = cost_init[:, 0].sum()
...
print(f"Cost reduction: {cost_init_total:.6f} -> {cost.sum():.6f} "
      f"({100 * (1 - cost.sum() / cost_init_total):.2f}% lower)")
```

Run directly with `python run_todorov_demo.py`. As with §4's Step 7,
this is the step that actually exercises your plant through the real
solver rather than just checking its contract (§6) — for this
reference system it should converge in well under 50 iterations with a
substantial cost reduction; if it doesn't, or `stop_reason` reports a
failure, recheck `derivs_fn` before suspecting the solver
(`test_ilqg.py` verifies `ilqg()` itself against an independently-
derived LQR solution, so a divergence on your own plant almost always
traces back to your derivatives, not the solver).

## 6. Validate before you run: `validate_plugin.py`

Both layers ship a validator: `core/ddp_solver/validate_plugin.py`
(`validate_core_plugin`) for `step_fn`/`derivs_fn`[/`constraint_fn`],
and `extensions/dual_control/validate_plugin.py`
(`validate_dual_control_plugin`) for
`dynamics`/`measurement`/`cost`/`continuous_dynamics`. Both are
**entirely opt-in** — nothing in the solver calls them automatically,
and adding them doesn't modify `ilqg.py`/`forward_pass.py`/
`back_pass.py`/`main_outer_control_loop.py` in any way. You call one of
them yourself, once, before wiring your functions into the real solver
(§4 Step 6, §5 Steps 4–5).

### What it checks

Per callable, the validator:

1. **Checks arity without calling your function** (`check_arity`) —
   confirms it accepts the documented number of positional arguments,
   so a plain signature mismatch is reported as one clear message
   instead of a `TypeError` raised from deep inside a real call three
   functions later.
2. **Calls your function with synthetic random data and catches any
   exception** (`safe_call`) — a bug that crashes your function becomes
   a reported issue (exception type + message preserved), not an
   aborted validation run that hides every other check that hasn't run
   yet.
3. **Checks the shape of every return value** (`check_shape`) against
   what its Protocol documents, and **checks for `NaN` leakage**
   (`check_no_nan`) on finite, in-range synthetic inputs.
4. **Checks the documented-convention gotchas explicitly**, not just
   shapes — specifically:
   - `noise_index=False` vs. `noise_index=0` must return the identical
     result (§3.2's truthy-check gotcha) — `dynamics`/`measurement`.
   - `w_from_filter=0` vs. `w_from_filter=1` must agree at zero noise
     (§3.2's two-shape-convention gotcha) — `dynamics`/`measurement`.
   - A `noise_index` request for column `i` returns the right shape
     (the noise-diffusion column, tiled across `K`) —
     `dynamics`/`measurement`.
   - A control column of all-`NaN` (the documented terminal marker)
     must not leak `NaN` into the returned cost — `cost`/`step_fn`,
     called out explicitly as **"the single most common plug-in
     bug"**.
   - `fxx`/`fxu`/`fuu` must be uniformly `None` or uniformly provided,
     never a mix — `derivs_fn`.
   - A per-timestep `lims` array from `constraint_fn` must never have
     `lower > upper` at any (timestep, control) position —
     `constraint_fn`.

Every check is run against **both** `augment_states` modes (`True` and
`False`) for the dual-control layer's four functions by default, since
§3.2 requires your functions to handle both — a plugin that only works
in one mode fails validation instead of only failing later, at
whichever call site happens to use the other mode.

### How to call it

Core layer:

```python
from core.ddp_solver.validate_plugin import validate_core_plugin

validate_core_plugin(step_fn, derivs_fn, n=n, m=m, N=N)
# constraint_fn is optional:
# validate_core_plugin(step_fn, derivs_fn, n=n, m=m, N=N, constraint_fn=my_constraint_fn)
```

Dual-control layer:

```python
from extensions.dual_control.validate_plugin import validate_dual_control_plugin

validate_dual_control_plugin(
    dynamics, measurement, cost, continuous_dynamics,
    nx=nx, nu=nu, nv=nv, ny=ny, constants=my_constants, n_p=n_p)
```

`constants` must be a **real constants object for your plant** (not
synthesized generically — the validator has no way to know what fields
it needs), shared across all four calls, exactly like `make_constants()`
in §4's Step 6.

### Reading the failure

Both entry points raise a single `PluginContractError` if anything
fails, carrying **every** issue found across every callable — not just
the first one hit — so you fix everything in one pass instead of
re-running the validator after each individual fix:

```
5 plug-in contract violation(s) found:
  - [dynamics] noise_index False vs 0 consistency (augment_states=True): dynamics(..., noise_index=False, ...) and dynamics(..., noise_index=0, ...) returned different results -- ...
  - [cost] terminal call, u[:, 0]=NaN (augment_states=True): cost contains NaN when a control column is the documented all-NaN terminal marker -- ...
  ...
```

Each line names the function (`[dynamics]`), which specific synthetic
call exposed it (`noise_index False vs 0 consistency
(augment_states=True)`), and what went wrong — the message text is
written to explain the *documented convention being violated*, not
just "assertion failed," so you can fix the actual bug without
re-reading the Protocol docstring from scratch. See §7 for a
per-mistake walkthrough of what each of these messages actually means
and how to fix it.

**What this does *not* check** (§4 Step 5's lesson, worth repeating
here): shape/signature/NaN-hygiene compliance is necessary, not
sufficient. A `cost` function that's perfectly contract-compliant can
still be economically or physically nonsensical (the lagged-state
pricing bug in §4 Step 7 passed this validator cleanly on both the
broken and fixed versions). Passing validation only means the solver
won't misuse your function — it says nothing about whether your
plant's actual behavior makes sense; that's what §4 Step 7 / §5 Step
5's end-to-end sanity check is for.

## 7. Common mistakes, and how the validator catches them

Ranked roughly by how often they actually happen, based on the bugs
found while building the two worked examples in this guide.

1. **Branching on `noise_index is False` instead of truthiness.**
   Different call sites in this codebase pass `noise_index` as `False`
   *and* as literal `0` to mean the same thing ("no specific noise
   column requested"). If your `dynamics`/`measurement` does
   `if noise_index is False:` instead of `if noise_index:`, it silently
   mishandles every call site that passes `0` — usually not the one you
   tested first. **Caught by:** the validator's "noise_index False vs 0
   consistency" check (§6), which calls your function both ways and
   confirms identical output.

2. **Only handling one `w_from_filter`/`v_from_filter` convention.**
   `w` (or `v`) is a genuinely different shape and meaning depending on
   this flag — a raw noise sample to scale yourself (`(nw, K)`) vs. an
   already-final perturbation to add directly (`(nxa, K)`). Code that
   assumes only one convention works fine wherever that convention is
   used and silently produces the wrong next state everywhere else.
   **Caught by:** the validator's "`w_from_filter` 0 vs 1 consistency at
   w=0" check, which confirms both conventions agree at zero noise.

3. **Not zeroing the all-`NaN` terminal control column before using
   it.** Both layers' cost functions (`cost` in the dual-control layer,
   the cost half of `step_fn` in the core layer) are called at a
   terminal step with `u` arriving as all-`NaN`, to mean "score
   final-state cost only." Using `u` directly at that point produces
   `NaN` cost, which then corrupts every downstream quadratic-cost-to-go
   computation in `back_pass.py`. This is called out in both Protocol
   docstrings as **the single most common plug-in bug**. **Caught by:**
   the validator's dedicated terminal-`NaN` check on every
   cost-producing function.

4. **Only implemented for one `augment_states` mode.** A plugin author
   who only ever tests with `augment_states=True` (the common case,
   since that's what online estimation needs) can leave the
   `augment_states=False` branch broken or missing (e.g. forgetting the
   `xa.shape[0] > 1 else np.full(K, constants.b_prior)` fallback pattern
   in §4 Step 5's `cost.py`). **Caught by:** every dual-control
   validator function runs both modes by default (§6).

5. **Mixing `fxx`/`fxu`/`fuu`** — providing some but not all of the
   second-order dynamics terms. `back_pass.py` treats "all `None`"
   (iLQG) and "all provided" (full DDP) as the only two valid states;
   anything else is an unhandled mix. **Caught by:**
   `validate_derivs_fn`'s explicit all-or-nothing check (§6).

6. **Writing code that only works for a specific `K`.** Since `K`
   varies call to call (§0's glossary: line-search step count one
   moment, sigma-point count the next), anything relying on
   `.squeeze()` to drop a size-1 batch axis, or a Python loop assuming a
   fixed column count, breaks the first time the solver calls your
   function with a different `K` than you happened to test. **Caught
   by:** the validator always tests with `K=3` — a `K=1`-only bug
   passes a naive smoke test but fails validation here.

7. **Confusing the two layers' `N`.** In `core/ddp_solver`, `N` =
   number of control steps (trajectory has `N+1` points). In
   `extensions/dual_control`, `N = round(T/dt)` counts sessions, and
   `u` there has `N-1` columns — a genuinely different relationship.
   Carrying an assumption from one layer's `N` into the other's
   silently misindexes a trajectory. **Not directly caught by the
   validator** (it's a documentation/convention issue, not a shape
   mismatch it can detect on its own) — re-read §3.1/§3.3's
   naming-gotcha notes if a trajectory-length-related off-by-one shows
   up.

8. **A `u_lim_method=2` tanh-squash saturation trap.** Not a contract
   violation at all — your function can be perfectly shape-correct and
   still get stuck because a badly-wrong initial belief pushes the raw
   pre-tanh control into `tanh`'s saturated region, where gradients
   vanish and warm-starting carries the stuck value forward session to
   session (§4 Step 7's second real bug). **Not caught by the
   validator** — this only shows up by actually running the controller
   end-to-end and watching whether chosen actions stay pinned at a
   bound (§4 Step 7 / §5 Step 5). If a controller won't move off a
   bound no matter how you retune the cost, try `u_lim_method=1` before
   suspecting the cost function.

## 8. Quick-reference checklist

- [ ] `pip install -e .` run from `PYTHON/` (§0) — otherwise every
      import below fails with `ModuleNotFoundError`.
- [ ] Picked the right layer: `core/ddp_solver/` if you don't need
      noise/online estimation, `extensions/dual_control/` if you do
      (§1).
- [ ] Read the relevant `protocols.py` for your chosen layer before
      writing any function (§2).
- [ ] Created `applications/<my_plant>/__init__.py` (§0).
- [ ] Wrote a constants dataclass carrying every plant-specific numeric
      parameter (dual-control layer; optional/closure-based for the
      core layer) (§4 Step 1 / §5 Step 1).
- [ ] `continuous_dynamics` (dual-control layer only) returns the
      documented 5-tuple, with `Fp` shaped `(0, 0)` when not augmented
      (§3.2, §4 Step 2).
- [ ] `dynamics`/`step_fn` branches on `noise_index` with
      `if noise_index:` (truthy), never `is False` (§3.2, §7.1).
- [ ] `dynamics`/`measurement` handles **both**
      `w_from_filter`/`v_from_filter` conventions and agrees between
      them at zero noise (§3.2, §7.2).
- [ ] `cost`/`step_fn` detects and zeros the all-`NaN` terminal control
      column **before** using it (§3.2, §3.3, §7.3).
- [ ] Every function handles **both** `augment_states=True` and
      `augment_states=False` (dual-control layer) (§3.1, §7.4).
- [ ] `derivs_fn`'s `fxx`/`fxu`/`fuu` are all `None` or all provided,
      never a mix (§3.3, §7.5).
- [ ] No code assumes a fixed `K` — vectorized over the batch axis, no
      `.squeeze()`-style size-1 assumptions (§3.1, §7.6).
- [ ] Ran `validate_core_plugin`/`validate_dual_control_plugin` and
      resolved every reported `PluginContractError` issue (§6).
- [ ] Ran the plant end-to-end through `ilqg()`/
      `main_outer_control_loop()` and sanity-checked the **actual
      behavior** (parameter estimate converges toward truth; chosen
      actions stay sensible, not pinned at a bound) — not just that
      validation passed (§4 Step 7, §5 Step 5, §7.8).

