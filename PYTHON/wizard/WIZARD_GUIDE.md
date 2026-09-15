# Wizard Guide: Building a Model Through Guided Questions

This guide is for someone who wants to define and test a model in this
framework **without writing any Python** — through a Streamlit app that
asks plain-language questions and runs the real solver for you. It
assumes nothing about this codebase, sympy, or control theory.

If you're comfortable writing Python and want to hand-write a plant
directly against `core/ddp_solver/protocols.py`/
`extensions/dual_control/protocols.py` instead, see
[`PLUGIN_GUIDE.md`](../PLUGIN_GUIDE.md) one level up — that's the raw-code
path this wizard is an alternative front end for. The two are related:
§7 below ("Graduating off the wizard") explains how to move from one to
the other.

---

## Table of contents

0. [One-time setup](#0-one-time-setup)
1. [What this wizard does](#1-what-this-wizard-does)
   - [1.1 Which setup do I need?](#11-which-setup-do-i-need)
2. [Walkthrough: building a model](#2-walkthrough-building-a-model)
3. [The saved model file](#3-the-saved-model-file)
4. [Writing equations](#4-writing-equations)
5. [What "Test This Model" checks](#5-what-test-this-model-checks)
6. [Reading your results](#6-reading-your-results)
7. [Graduating off the wizard](#7-graduating-off-the-wizard)
8. [Troubleshooting common messages](#8-troubleshooting-common-messages)

---

## 0. One-time setup

From `PYTHON/` (run once):

```
pip install -e .
pip install streamlit sympy pandas matplotlib
```

Then launch the wizard from `PYTHON/`:

```
streamlit run wizard/app.py
```

This opens the app in your browser (usually `http://localhost:8501`).
Everything you enter is saved to `wizard/models/<your model name>.json`
when you click **Save model** or **Save now** — nothing is lost if you
close the browser tab, as long as you saved first.

---

## 1. What this wizard does

You answer questions about your system in plain language — what
quantities it tracks ("states"), what you can control ("actions"),
what's unknown and should be learned from data ("parameters"), whether
any control needs a state-dependent limit ("constraints"), how things
change over time ("dynamics"), what can be measured ("measurement"), and
what you're trying to optimize ("cost"). The wizard turns those answers
into a real model, checks it for problems, and runs one of two real
solvers already used elsewhere in this project — which one is a single
choice you make in Advanced settings (§1.1):

- **iLQG / Dual Control** (`extensions/dual_control/main_outer_control_loop`)
  — the exact machinery behind `applications/dynamic_pricing/` and
  `applications/duopoly_pricing/`. Estimates unknown parameters online
  while controlling (MPC replanning every step).
- **iLQR** (`core/ddp_solver/ilqg`, via `wizard/core_ilqr_adapter.py`) —
  one deterministic, full-horizon solve, no online estimation. The same
  solver behind `applications/robot_arm_2link/` and
  `applications/todorov_toy/`.

Either way, you're reachable without writing any of that machinery's
code by hand.

The wizard works identically regardless of what your system actually
is — a rehab plan, a pricing problem, a physical process, anything. It
never assumes a domain.

### 1.1 Which setup do I need?

Same two independent choices as the hand-written path
([`PLUGIN_GUIDE.md`](../PLUGIN_GUIDE.md)'s §1.1) — the wizard just asks
them through the UI instead of file-writing:

**Choice 1 — Control method** (step 10, "Advanced settings" — a radio
button, default `iLQG / Dual Control`):

| You need... | Pick | What happens |
|---|---|---|
| Online parameter estimation, noise, dual control ("curiosity") | `iLQG / Dual Control` | Step 4's Unknown parameters are estimated from data as the model runs (MPC replanning every session) |
| Just the best deterministic open-loop trajectory, no unknowns | `iLQR` | Step 4's Unknown parameters (if any) are used at their fixed prior/best-guess value only — never learned; needs at least 2 steps to simulate (Basics, step 1) |

Switching this radio button doesn't hide step 4 — if you leave
parameters defined while `iLQR` is selected, the Unknown parameters step
shows an inline note that they'll be treated as fixed constants instead
of learned.

**Choice 2 — Constraints** (step 5, optional, works with *either*
control method): only fill this in if a control needs a bound that
*shrinks or shifts depending on the current state* — a plain action
min/max (step 3) can't express that. Each row is one inequality
`h(...) >= 0`, of one of two kinds:

- **`h(x, u) >= 0`** — may reference states, actions, and constants.
- **`h(x) >= 0`** — states and constants only (no actions); reduced
  internally via a Lie derivative + the row's `alpha` value before being
  enforced the same way as the first kind.

Both kinds only actually restrict anything if they (or, for `h(x)`, its
Lie derivative) depend on **exactly one** action — Test This Model (step
12) warns you if a constraint you wrote turns out not to (rather than
silently doing nothing), and raises a hard error if it depends on two or
more. Enabling any constraint requires "Bound enforcement method" (step
10) to stay at its default, `Box-QP` — the alternative (`Tanh-squash`)
doesn't compose with a state-dependent bound.

**So, four combinations exist**, same as the hand-written path:

1. iLQG / Dual Control, no constraints — the two shipped examples in §2
   (`pricing_example.json`, `duopoly_example.json`), and
   `wizard/models/2_Link_Robotic_Arm.json` as saved (learns an unknown
   link mass, no constraint attached).
2. iLQG / Dual Control + constraints — no shipped example ships with one
   saved, but this is a real, verified combination: load
   `wizard/models/2_Link_Robotic_Arm.json`, add one `state_action`
   constraint via step 5 (e.g. a speed-derated torque cap on one joint),
   and Run — parameter learning and the constraint operate together on
   the same model (the concrete case that motivated adding constraints
   to this control method at all).
3. iLQR, no constraints — pick `iLQR` in step 10 with step 5 empty.
4. iLQR + constraints — see the shipped
   `wizard/examples/robot_arm_ilqr_constrained.json`.

---

## 2. Walkthrough: building a model

Open the app and choose **Start a new model**, **Load a saved model**, or
**Load an example**. The two shipped examples
(`wizard/examples/pricing_example.json` and
`wizard/examples/duopoly_example.json`) are good first stops — load one
and click through the steps to see a complete, working answer set before
building your own.

The steps, in order (use the sidebar to jump between them at any time —
nothing is locked until you reach Validate/Run):

1. **Basics** — name, description, how much real time one step
   represents (`dt`), and how many steps to simulate (iLQR needs at
   least 2; see §1.1).
2. **States** — the quantities your system tracks over time. Give each
   one a name, a starting value, and (optionally) a valid range it
   should be clamped to.
3. **Actions** — what you can choose or control at each step. Each one
   needs a minimum and maximum allowed value.
4. **Unknown parameters** — anything whose true value you don't know but
   want the controller to learn as it runs. It's fine to have none. Only
   actually *learned* under the `iLQG / Dual Control` control method
   (step 10, §1.1) — under `iLQR` a parameter defined here is used at
   its fixed prior value instead, and the step shows an inline note
   saying so. If you're testing in simulation rather than on real data,
   you can supply a "true value" here so the results show whether the
   controller learned correctly.
5. **Constraints** (optional) — require `h(...) >= 0` to hold throughout
   the run, enforced through the existing constrained-DDP extension
   (Dastan & Sensinger 2024). It's fine to have none — action bounds
   (step 3) always apply regardless of what's here. Works with either
   control method (§1.1); requires "Bound enforcement method" (step 10)
   to stay at `Box-QP`.
6. **Constants** — any other fixed, known numbers your equations need.
7. **Dynamics** — one equation per state, describing its rate of change
   in terms of your states/actions/parameters/constants. See §4.
8. **Measurement** — what can actually be observed. The simplest case is
   just writing a state's name to observe it directly.
9. **Cost** — an expression that's small when things are going well and
   large when they're going badly. This is what the controller minimizes.
10. **Advanced settings** — solver internals with sensible defaults
    already filled in, plus the **Control method** choice (§1.1: `iLQG /
    Dual Control` vs. `iLQR`). The bound-enforcement method is
    recommended to stay at its default (box-QP) — the alternative can
    trap the controller when the initial guess is very wrong (a real
    failure mode found while building the two example models this
    wizard is based on), and is also incompatible with any enabled
    constraint from step 5.
11. **Review & Save** — see the full configuration, a preview of what it
    looks like as Python source (see §7), and save it to disk.
12. **Validate** — click **Test This Model** to check your equations and
    settings against sample numbers before anything touches the real
    solver. This must pass before you can run a simulation.
13. **Run & Results** — click **Run Simulation** to run the real solver
    and see plots and a results table. Under `iLQR`, also shows the
    optimizer's stop reason and a cost/gradient-norm convergence plot
    (§6) — the `iLQG / Dual Control` MPC loop has no single "converged"
    moment to plot, so that part doesn't appear there.

---

## 3. The saved model file

Every "Save" writes one JSON file to `wizard/models/<name>.json`. It's
plain, readable JSON — you can open it in a text editor, back it up,
version-control it, or hand-edit it (reload it in the wizard afterward to
re-check it). Its top-level shape:

```jsonc
{
  "name": "...", "description": "...", "dt": 1.0, "n_sessions": 15,
  "control_method": "ilqg",
  "states": [{"name": "...", "initial_value": 0.0, "min": null, "max": null, "process_noise": 1.0}],
  "actions": [{"name": "...", "min": 0.0, "max": 1.0}],
  "parameters": [{"name": "...", "prior_guess": 0.0, "prior_variance": 1.0, "process_noise": 0.05, "true_value": null}],
  "constants": [{"name": "...", "value": 0.0}],
  "dynamics": {"<state name>": "<equation>"},
  "measurement": [{"name": "...", "expression": "...", "noise_scale": 1.0}],
  "cost": {"running": "...", "terminal": "0"},
  "constraints": [{"name": "...", "kind": "state_action", "expression": "...", "alpha": 1.0, "enabled": true}],
  "solver": {"u_lim_method": 1, "reg_type": 1, "max_du_iterations": 60, "first_run_max_du_iterations": 100, "seed": 0}
}
```

`control_method` is `"ilqg"` (default) or `"ilqr"` (§1.1) — a file saved
before this field existed simply omits it, and loading it back defaults
to `"ilqg"`, so old saved models keep working unchanged. `constraints`
similarly defaults to an empty list when omitted; `kind` is
`"state_action"` (`h(x, u) >= 0`) or `"state_only"` (`h(x) >= 0`, uses
`alpha`).

(See `wizard/schema.py`'s `ModelConfig` for the exact, authoritative
field list — this is a summary, not a spec.)

---

## 4. Writing equations

Equations appear in three places: **Dynamics** (one per state), each
**Measurement** row's expression, and **Cost**'s running/terminal fields.
All three use the same rules:

- Reference any name you've already defined (a state, action, parameter,
  or constant) by typing it exactly as you named it.
- Allowed functions: `sin, cos, tan, exp, log, sqrt, tanh, abs, min, max`.
- Write multiplication explicitly: `2*x1`, not `2x1`.
- Use `^` or `**` for powers.
- Decimal numbers work as expected: `3.14`.

Every equation box gives live feedback as you type (or right after you
finish typing) — a green "Looks good" or a plain-language explanation of
what's wrong, with a suggestion if it looks like a typo. This check runs
the same underlying parser that later compiles your model — if it says
"Looks good" here, that part will parse cleanly at Validate time too.

**On unknown parameters in Cost:** you can reference a parameter's name
directly in your cost expression — the wizard automatically uses the
controller's current best estimate of it. A tip shown on the Cost step is
worth reading if your action affects your system immediately rather than
with a delay: writing the cost in terms of that immediate effect (rather
than only a state that reacts with a lag) helps the controller actually
respond to the current action, not just future consequences of it.

---

## 5. What "Test This Model" checks

Clicking **Test This Model** runs two layers of checking, both before
anything touches the real solver:

1. **Structural checks** — every name is unique and valid, every state
   has exactly one dynamics equation, action bounds are in the right
   order, noise scales aren't negative, `control_method` is a known
   value, `n_sessions >= 2` under iLQR, every constraint has a unique
   name/valid kind/non-empty expression (and a positive `alpha` if it's
   an `h(x)` constraint), and an enabled constraint requires
   "Bound enforcement method" = Box-QP, and so on.
2. **Contract checks** — your equations are compiled into real callables
   and run against sample numbers, using whichever validator matches
   your Control method (step 10): `extensions/dual_control/
   validate_plugin.validate_dual_control_plugin` for `iLQG / Dual
   Control`, or `core/ddp_solver/validate_plugin.validate_core_plugin`
   for `iLQR` — the SAME validators every hand-written plant in this
   codebase is checked against (`applications/dynamic_pricing/`,
   `applications/duopoly_pricing/`, `applications/robot_arm_2link/`,
   `applications/todorov_toy/`, etc.). Both catch shape mismatches, NaN
   leaks, and the most common plug-in bug (a cost/dynamics function that
   doesn't handle the "final step" marker correctly) — the wizard's
   generated code handles that one automatically, so you'll only see it
   if something is unexpectedly malformed. If step 5 has an enabled
   constraint, it's compiled and checked here too (`validate_constraint_fn`)
   — including a check that surfaces, as a non-fatal warning rather than
   letting it pass silently, the case where a constraint you wrote turns
   out not to actually depend on any single action (§1.1).

Any problem found is shown as plain language, never a raw Python
traceback. If something truly unexpected happens, a **Technical details**
panel below the message has the underlying error, for anyone helping you
debug it.

---

## 6. Reading your results

After a successful run you'll see:

- **State plots** — one per state, true value and (if applicable)
  the controller's estimate of it, over each step.
- **Action plots** — what was chosen at each step, with the allowed
  range marked.
- **Parameter plots** (only if you defined unknown parameters) — the
  estimate over time, with the initial guess and (if you supplied one)
  the true value marked for comparison.
- **Cost plots** — cost per step and cumulative cost.
- **A results table** — one row per step, columns named after your own
  state/action/parameter names — downloadable as CSV.
- **Optimizer readout + convergence plot** (`iLQR` control method only,
  §1.1/§2 step 13) — the stop reason (e.g. "SUCCESS: gradient norm <
  tol_grad") and a cost-vs-iteration / gradient-norm-vs-iteration plot
  from the single full-horizon solve. A non-"SUCCESS" stop reason
  doesn't necessarily mean the result is useless — a constrained problem
  in particular can make a real, still-usable trajectory stop early for
  a benign reason; check the actual state/action/cost plots too, not
  just this readout. There's no equivalent for `iLQG / Dual Control` —
  its MPC loop re-plans every session, so there's no single "converged"
  moment to plot.

If a parameter's estimate doesn't fully converge to its true value,
that's not necessarily a bug in your model — some parameter combinations
are only weakly identifiable from the data your system's dynamics
actually produce (this is a real, well-understood phenomenon, not unique
to the wizard — see `applications/duopoly_pricing/`'s own documented
findings on this in the main `CLAUDE.md`, which this wizard's harder
shipped example reproduces faithfully).

---

## 7. Graduating off the wizard

If a model outgrows the wizard's question flow (you need dynamics or a
cost structure genuinely too complex to express as a single equation per
state, for example), the **Review & Save** step's "View generated code"
panel gives you a starting point: a read-only preview of your model as
Python source, in the same style as
`applications/dynamic_pricing/`/`applications/duopoly_pricing/`
(`iLQG / Dual Control` models) or `applications/robot_arm_2link/`
(`iLQR` models — see §1.1 for which control method your model uses).
Copy it into a new `applications/<your model>/` folder and follow
[`PLUGIN_GUIDE.md`](../PLUGIN_GUIDE.md) from there — particularly its
§1.1 (confirms which layer you're graduating into), §3 (the exact
contract for that layer), and §6 (the validator) — to finish it by hand.
If step 5 has any constraints, `PLUGIN_GUIDE.md`'s `constraint_fn`
subsections (§3.2 for the dual-control layer, §3.3 for the core layer)
cover the hand-written equivalent of what `wizard/core_ilqr_adapter.py`'s
`build_constraint_fn` compiles for you here.

This preview is for reading and copying only — the wizard itself never
executes it. Your model runs through a single, thoroughly-tested generic
engine (`wizard/generic_plant.py` for the plant callables, plus
`wizard/core_ilqr_adapter.py` for the `iLQR` control method's
`step_fn`/`derivs_fn`/`constraint_fn` wrapping) built directly from your
parsed equations, not by running this generated text.

---

## 8. Troubleshooting common messages

| Message | What it means |
|---|---|
| "I don't recognize the name '...' in this equation" | A typo, or you're referencing a state/action/parameter/constant you haven't defined yet (check the "Known names" line above the equation box). |
| "I don't recognize the function '...(...)' in this equation" | Only `sin, cos, tan, exp, log, sqrt, tanh, abs, min, max` are supported. |
| "Couldn't understand this equation... missing '*' for multiplication" | Write `2*x1`, not `2x1` — multiplication is never implicit. |
| "This equation uses character(s) that aren't allowed" | Only letters, digits, underscores, spaces, and `+ - * / ( ) . , ^` are permitted in an equation. |
| "the minimum must be less than the maximum" | An action's or state's min/max bounds are reversed or equal. |
| "Problem found in the dynamics equations / cost expression / measurement expressions (...)" | A contract check failed on sample numbers — see §5; read the message's specific check name for what went wrong. |
| "This run took longer than N seconds and was stopped" | Try lowering the number of steps to simulate, or the planning-iteration limits under Advanced settings. |
| "The solver produced invalid numbers (NaN/Inf)" | Usually an equation divides by something that can reach zero, or a noise/bound value is too extreme — check equations flagged with a division warning first. |
| "Please pass validation (step 12) first" | The **Run Simulation** button is intentionally disabled until your current model has passed **Test This Model** — this can't be bypassed. |
| "iLQR control method needs at least 2 steps to simulate" | Raise the number of steps to simulate (step 1) — iLQR needs at least one state to start from plus at least one control step; this doesn't apply under `iLQG / Dual Control`. |
| "Control method is currently iLQR..., any parameter defined here is used at its fixed 'best initial guess' value instead of being learned" | Informational, shown on step 4 — not an error. Switch Control method (step 10) to `iLQG / Dual Control` if you actually want this parameter estimated from data. |
| "Constraints require the 'Box-QP' bound enforcement method" | An enabled constraint (step 5) needs "Bound enforcement method" (step 10) set to `Box-QP` — the `Tanh-squash` alternative doesn't compose with a state-dependent bound; switch it, or disable/remove the constraint. |
| "alpha must be a positive number for a state-only constraint" | Every `h(x) >= 0` constraint (step 5) needs `alpha > 0` — it controls how fast the constraint's internal reduction term decays. `h(x, u) >= 0` constraints ignore `alpha` entirely. |
| A constraint warning at Validate time saying it doesn't actually depend on any single action | Non-fatal (§1.1, §5) — the constraint compiles and runs, but has no effect, because it (or, for `h(x)`, its Lie derivative) doesn't depend on exactly one action. Rewrite the expression so it does, or remove the constraint. |
| A plugin-contract error mentioning "depends on 2 control channels" (or more) | A constraint (step 5) depends on more than one action — not expressible as a single control's box bound under this constraint method. Split it, rewrite it to isolate one action, or remove it. |
