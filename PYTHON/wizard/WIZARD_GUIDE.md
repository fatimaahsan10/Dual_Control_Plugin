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
what's unknown and should be learned from data ("parameters"), how
things change over time ("dynamics"), what can be measured
("measurement"), and what you're trying to optimize ("cost"). The wizard
turns those answers into a real model, checks it for problems, and runs
the same dual-control solver used everywhere else in this project
(`extensions/dual_control/main_outer_control_loop`) — the exact machinery
behind `applications/dynamic_pricing/` and
`applications/duopoly_pricing/`, just reachable without writing any of
their code by hand.

The wizard works identically regardless of what your system actually
is — a rehab plan, a pricing problem, a physical process, anything. It
never assumes a domain.

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
   represents (`dt`), and how many steps to simulate.
2. **States** — the quantities your system tracks over time. Give each
   one a name, a starting value, and (optionally) a valid range it
   should be clamped to.
3. **Actions** — what you can choose or control at each step. Each one
   needs a minimum and maximum allowed value.
4. **Unknown parameters** — anything whose true value you don't know but
   want the controller to learn as it runs. It's fine to have none. If
   you're testing in simulation rather than on real data, you can supply
   a "true value" here so the results show whether the controller learned
   correctly.
5. **Constants** — any other fixed, known numbers your equations need.
6. **Dynamics** — one equation per state, describing its rate of change
   in terms of your states/actions/parameters/constants. See §4.
7. **Measurement** — what can actually be observed. The simplest case is
   just writing a state's name to observe it directly.
8. **Cost** — an expression that's small when things are going well and
   large when they're going badly. This is what the controller minimizes.
9. **Advanced settings** — solver internals with sensible defaults
   already filled in. The bound-enforcement method is recommended to
   stay at its default (box-QP) — the alternative can trap the controller
   when the initial guess is very wrong (a real failure mode found while
   building the two example models this wizard is based on).
10. **Review & Save** — see the full configuration, a preview of what it
    looks like as Python source (see §7), and save it to disk.
11. **Validate** — click **Test This Model** to check your equations and
    settings against sample numbers before anything touches the real
    solver. This must pass before you can run a simulation.
12. **Run & Results** — click **Run Simulation** to run the real solver
    and see plots and a results table.

---

## 3. The saved model file

Every "Save" writes one JSON file to `wizard/models/<name>.json`. It's
plain, readable JSON — you can open it in a text editor, back it up,
version-control it, or hand-edit it (reload it in the wizard afterward to
re-check it). Its top-level shape:

```jsonc
{
  "name": "...", "description": "...", "dt": 1.0, "n_sessions": 15,
  "states": [{"name": "...", "initial_value": 0.0, "min": null, "max": null, "process_noise": 1.0}],
  "actions": [{"name": "...", "min": 0.0, "max": 1.0}],
  "parameters": [{"name": "...", "prior_guess": 0.0, "prior_variance": 1.0, "process_noise": 0.05, "true_value": null}],
  "constants": [{"name": "...", "value": 0.0}],
  "dynamics": {"<state name>": "<equation>"},
  "measurement": [{"name": "...", "expression": "...", "noise_scale": 1.0}],
  "cost": {"running": "...", "terminal": "0"},
  "solver": {"u_lim_method": 1, "reg_type": 1, "max_du_iterations": 60, "first_run_max_du_iterations": 100, "seed": 0}
}
```

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
   order, noise scales aren't negative, and so on.
2. **Contract checks** — your equations are compiled into real callables
   and run against sample numbers using
   `extensions/dual_control/validate_plugin.validate_dual_control_plugin`
   — the SAME validator every hand-written plant in this codebase is
   checked against (`applications/dynamic_pricing/`,
   `applications/duopoly_pricing/`, etc.). It catches shape mismatches,
   NaN leaks, and the most common plug-in bug (a cost/dynamics function
   that doesn't handle the "final step" marker correctly) — the wizard's
   generated code handles that one automatically, so you'll only see it
   if something is unexpectedly malformed.

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
`applications/dynamic_pricing/`/`applications/duopoly_pricing/`. Copy it
into a new `applications/<your model>/` folder and follow
[`PLUGIN_GUIDE.md`](../PLUGIN_GUIDE.md) from there — particularly its
§3 (the exact contract) and §6 (the validator) — to finish it by hand.

This preview is for reading and copying only — the wizard itself never
executes it. Your model runs through a single, thoroughly-tested generic
engine (`wizard/generic_plant.py`) built directly from your parsed
equations, not by running this generated text.

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
| "Please pass validation (step 11) first" | The **Run Simulation** button is intentionally disabled until your current model has passed **Test This Model** — this can't be bypassed. |
