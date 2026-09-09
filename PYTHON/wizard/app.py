"""
Streamlit entry point for the generic model-building wizard. Run with:

    streamlit run wizard/app.py

(from PYTHON/, with `pip install -e .` already done per PLUGIN_GUIDE.md
§0 -- see WIZARD_GUIDE.md for the guided walkthrough).

This file is UI ORCHESTRATION ONLY -- every piece of actual logic
(parsing, compiling, validating, running, plotting) lives in the other
wizard/*.py modules, each independently unit-tested. app.py's own job is
just: collect answers into a schema.ModelConfig via Streamlit widgets,
and call those modules at the right time, wrapped so that nothing here
can crash the session (see the module-level `try/except` around every
step's render call, and every button handler's own try/except around
calls that can raise a caller-facing exception -- storage.StorageError
being the only one; validation_runner/solver_runner/plotting are already
internally exception-safe and return report objects instead of raising).

Session-state keys used (see `_init_state`):
    config                    -- the ModelConfig being built, or None
                                  before "Start" is completed
    step_idx                  -- 0-based index into STEP_NAMES; the
                                  sidebar radio's OWN widget key (see
                                  `_goto` for why this must be the single
                                  source of truth, not a separately
                                  tracked variable)
    validation_report         -- last ValidationReport, or None
    validated_config_signature -- config signature validation_report was
                                  computed against, so a config edited
                                  after validating can't silently reuse a
                                  stale "passed" result to unlock Run
    solver_report              -- last SolverReport, or None
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Community Cloud runs `streamlit run PYTHON/wizard/app.py` directly,
# without our local `pip install -e .` step (see PYTHON/pyproject.toml), and
# Streamlit only puts this file's own directory (wizard/) on sys.path -- not
# its parent PYTHON/, which is what the `wizard.*`/`core.*`/`extensions.*`/
# `applications.*` absolute imports below actually need. Adding PYTHON/ here
# makes the imports resolve regardless of how/where this script is launched
# from, on top of (not instead of) the editable install local dev already
# uses -- harmless if PYTHON/ is already importable.
_PYTHON_DIR = str(Path(__file__).resolve().parent.parent)
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import json
from dataclasses import MISSING, asdict, dataclass

import numpy as np
import pandas as pd
import streamlit as st

from wizard.codegen_preview import generate_all_previews
from wizard.equation_parser import ALLOWED_FUNCTIONS, check_equation_live
from wizard.plotting import (
    plot_actions, plot_convergence, plot_cost, plot_parameters, plot_states,
    results_table,
)
from wizard.schema import (
    CONSTRAINT_KINDS, CONTROL_METHODS, MAX_DU_ITERATIONS,
    MAX_FIRST_RUN_DU_ITERATIONS, MAX_N_SESSIONS,
    ActionSpec, ConstantSpec, ConstraintSpec, MeasurementSpec, ModelConfig,
    ParameterSpec, StateSpec,
)
from wizard.solver_runner import run_solver
from wizard.storage import (
    StorageError, list_examples, list_models, load_model, load_model_from_path,
    save_model,
)
from wizard.validation_runner import run_validation

STEP_NAMES = [
    "1. Basics", "2. States", "3. Actions", "4. Unknown parameters",
    "5. Constraints", "6. Constants", "7. Dynamics", "8. Measurement",
    "9. Cost", "10. Advanced settings", "11. Review & Save", "12. Validate",
    "13. Run & Results",
]


def _init_state() -> None:
    st.session_state.setdefault("config", None)
    st.session_state.setdefault("step_idx", 0)
    st.session_state.setdefault("validation_report", None)
    st.session_state.setdefault("validated_config_signature", None)
    st.session_state.setdefault("solver_report", None)


# Step-2..7 tables built with _select_row/_row_form below; each keeps its
# own "which row is selected" widget key in session_state (see
# `_reset_row_editor_state`).
_ROW_TABLE_NAMES = ["states", "actions", "parameters", "constraints",
                    "constants", "measurement"]


def _reset_row_editor_state() -> None:
    """Drop every table's "which row is selected" widget state.

    Must be called any time `st.session_state.config` is replaced with a
    *different* model (new/loaded/reset) -- otherwise a table's selector
    could keep pointing at an index from the PREVIOUS model instead of
    resetting to "+ Add new" for the newly-loaded one."""
    for name in _ROW_TABLE_NAMES:
        st.session_state.pop(f"{name}_select", None)


def _config_signature(cfg: ModelConfig) -> str:
    return json.dumps(cfg.to_dict(), sort_keys=True, default=str)


def _goto(step_idx: int) -> None:
    """Programmatically jump to a step. Must set the widget's OWN
    session_state key (not a separate tracking variable) BEFORE that
    widget is instantiated this rerun -- Streamlit's supported pattern
    for setting a keyed widget's value from code; setting a different
    variable and passing it as `index=` to a keyed radio conflicts with
    the widget's own persisted state and is silently ignored on repeat
    runs."""
    st.session_state["step_idx"] = step_idx


def _show_equation_feedback(text: str, all_names: list[str]) -> None:
    if not text.strip():
        return
    ok, msg = check_equation_live(text, all_names)
    if ok:
        st.success("Looks good.", icon="✅")
    else:
        st.error(msg, icon="⚠️")


def _equation_syntax_caption(all_names: list[str]) -> None:
    """Shared "how do I write an equation here" help, shown above every
    equation-entry field (Dynamics/Measurement/Cost). Previously only
    Dynamics explained the syntax at all, and none of the three steps
    showed a worked example -- a real source of confusion reported by a
    user (equations look like a single free-text box with no visible
    hint of what's actually accepted)."""
    example = f"0.5*{all_names[0]} - sin({all_names[1]})" if len(all_names) >= 2 \
        else "0.5*x1 - sin(u1)" if not all_names \
        else f"0.5*{all_names[0]}^2"
    st.caption(
        f"Known names: {', '.join(all_names) or '(none defined yet)'}. "
        f"Write ordinary math: `+ - * / ^` (`^` is power) and parentheses, "
        f"decimal numbers like `3.14`, and these functions: "
        f"{', '.join(sorted(ALLOWED_FUNCTIONS))} (plus the constants `pi`, "
        f"`e`). **Multiplication must be written out** -- `2*x1`, never "
        f"`2x1`. Example: `{example}`")


# ----------------------------------------------------------------------
# Shared row-list editor for steps 2-5 and 7 (states/actions/parameters/
# constants/measurement) -- add/edit/delete ONE row at a time via plain
# widgets (text_input/number_input/selectbox/form), never st.data_editor.
# st.data_editor's canvas-based cell editor has a real, reproducible
# upstream bug (streamlit/streamlit#7354, #7749, #7868): pressing Enter
# right after typing into a cell sometimes silently reverts it instead of
# saving, even before any Streamlit rerun happens. Clicking away to blur
# the cell works, Enter does not -- confirmed live on the deployed app,
# confirmed independent of hosting (it's in Streamlit's bundled frontend
# code). Plain widgets don't have this bug.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class _Field:
    """One editable field in a _row_form. `kind` is "text", "float" (a
    plain required number), "optional_float" (a number_input that can
    be left blank, returning None -- for Optional[float] dataclass
    fields), "bool" (a checkbox), or "select" (a selectbox restricted to
    `options`, added for ConstraintSpec.kind -- see render_constraints)."""
    attr: str
    label: str
    kind: str
    min_value: float | None = None
    help: str | None = None
    options: tuple | None = None


def _select_row(items: list, key_prefix: str) -> int | None:
    """The "which row to edit, or add new" selector shared by every
    table step. Returns the selected row's index, or None for "+ Add
    new" -- index rather than name, so two rows that happen to share a
    name (only caught at cfg.validate() time, not blocked here) can't
    collide on the same selection."""
    def _label(i: int | None) -> str:
        if i is None:
            return "+ Add new"
        name = items[i].name or "(blank)"
        return f"{i}: {name}"

    return st.selectbox(
        "Edit existing or add new:", options=[None] + list(range(len(items))),
        format_func=_label, key=f"{key_prefix}_select")


def _row_form(
    cfg: ModelConfig, items: list, spec_cls: type, fields: list[_Field],
    key_prefix: str, selected_idx: int | None, *,
    on_change=None, extra_values: dict | None = None,
    extra_keys_to_clear: list[str] | None = None,
) -> None:
    """Renders the add/edit form for `items[selected_idx]` (or a blank
    "add new" row if `selected_idx` is None), plus a read-only overview
    table of every current row below it. Paired with `_select_row`.

    `on_change(cfg)`, if given, runs after a successful Save/Delete --
    e.g. states use it to prune cfg.dynamics of removed/renamed states.
    `extra_values` are merged into the constructed dataclass alongside
    the form's own fields -- e.g. measurement's `expression` field lives
    OUTSIDE this form (see render_measurement) so it can get live
    equation-syntax feedback per keystroke, which a field inside a form
    can't (forms only report values on submit). `extra_keys_to_clear`
    lets a caller ask for that kind of outside-form widget's session
    state to be reset too, alongside this form's own fields, whenever a
    Save/Delete resets the selector back to "+ Add new".

    IMPORTANT: "Save" must be the FIRST st.form_submit_button rendered
    below -- Streamlit's Enter-to-submit fires whichever submit button is
    first in the form, regardless of which field has focus. Declaring
    Delete first would make pressing Enter delete the row instead of
    saving it -- exactly the kind of Enter-key surprise this whole
    rewrite exists to get rid of."""
    current = items[selected_idx] if selected_idx is not None else None

    with st.form(key=f"{key_prefix}_form", clear_on_submit=True):
        values = {}
        for f in fields:
            if current is not None:
                default = getattr(current, f.attr)
            else:
                # required fields (currently just "name") have no
                # dataclass default -- MISSING, not a usable value.
                dc_default = spec_cls.__dataclass_fields__[f.attr].default
                default = None if dc_default is MISSING else dc_default
            widget_key = f"{key_prefix}_{selected_idx}_{f.attr}"
            if f.kind == "text":
                values[f.attr] = st.text_input(
                    f.label, value=default or "", key=widget_key, help=f.help)
            elif f.kind == "bool":
                values[f.attr] = st.checkbox(
                    f.label, value=bool(default), key=widget_key, help=f.help)
            elif f.kind == "select":
                opts = list(f.options)
                idx_ = opts.index(default) if default in opts else 0
                values[f.attr] = st.selectbox(
                    f.label, options=opts, index=idx_, key=widget_key, help=f.help)
            else:
                values[f.attr] = st.number_input(
                    f.label, value=default, min_value=f.min_value,
                    key=widget_key, help=f.help)

        save = st.form_submit_button("Save" if current is not None else "Add")
        delete = st.form_submit_button("Delete") if current is not None else False

    if save or delete:
        if save:
            name = values.get("name", "").strip()
            if not name:
                st.error("Please enter a name.")
                return
            values["name"] = name
            new_item = spec_cls(**{**values, **(extra_values or {})})
            if selected_idx is not None:
                items[selected_idx] = new_item
            else:
                items.append(new_item)
        else:
            del items[selected_idx]
        if on_change is not None:
            on_change(cfg)
        for k in (extra_keys_to_clear or []):
            st.session_state.pop(k, None)
        st.session_state.pop(f"{key_prefix}_select", None)
        st.rerun()

    if items:
        st.dataframe(pd.DataFrame([asdict(i) for i in items]),
                     use_container_width=True)
    else:
        st.caption("No rows yet.")


# ----------------------------------------------------------------------
# Start screen (shown when no model is loaded/created yet)
# ----------------------------------------------------------------------

def render_start() -> None:
    st.title("Model Wizard")
    st.write(
        "Build and test a control/estimation model through guided "
        "questions -- states, actions, unknown parameters, dynamics, "
        "measurement, and cost -- with no code required.")
    st.warning(
        "On the hosted (Streamlit Community Cloud) version of this app, "
        "'Save' only writes to that container's temporary disk -- it can "
        "be wiped by a restart (inactivity, redeploy, or a resource "
        "limit) and won't show up under 'Load a saved model' afterwards. "
        "**Use 'Download model file' (step 10) to save a copy to your "
        "own computer, and 'Upload a model file' below to resume from "
        "it** -- that's the only way progress reliably survives a "
        "restart on the hosted deployment.", icon="⚠️")

    choice = st.radio(
        "What would you like to do?",
        ["Start a new model", "Load a saved model", "Upload a model file",
         "Load an example"],
        key="start_choice")

    if choice == "Start a new model":
        name = st.text_input("Name your new model:", key="new_model_name")
        if st.button("Create", key="create_new"):
            if not name.strip():
                st.error("Please enter a name.")
            else:
                st.session_state.config = ModelConfig(name=name.strip())
                st.session_state.validation_report = None
                st.session_state.solver_report = None
                _reset_row_editor_state()
                _goto(0)
                st.rerun()

    elif choice == "Load a saved model":
        names = list_models()
        if not names:
            st.info("No saved models yet on this container's disk. If you "
                     "saved one earlier and it's gone, the container "
                     "likely restarted -- see the warning above.")
        else:
            sel = st.selectbox("Choose a saved model:", names, key="load_select")
            if st.button("Load", key="load_btn"):
                try:
                    st.session_state.config = load_model(sel)
                    st.session_state.validation_report = None
                    st.session_state.solver_report = None
                    _reset_row_editor_state()
                    _goto(0)
                    st.rerun()
                except StorageError as e:
                    st.error(e.message)

    elif choice == "Upload a model file":
        uploaded = st.file_uploader(
            "Choose a model .json file you previously downloaded:",
            type="json", key="upload_model_file")
        if uploaded is not None and st.button("Load uploaded file", key="load_uploaded_btn"):
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8"))
                st.session_state.config = ModelConfig.from_dict(data)
                st.session_state.validation_report = None
                st.session_state.solver_report = None
                _reset_row_editor_state()
                _goto(0)
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"That file isn't valid JSON ({e}).")
            except (TypeError, ValueError, KeyError) as e:
                st.error(
                    f"That file has missing or mismatched fields "
                    f"({type(e).__name__}: {e}) -- it may be from an "
                    f"incompatible version or have been hand-edited "
                    f"incorrectly.")

    else:
        examples = list_examples()
        if not examples:
            st.info("No example models are shipped with this installation.")
        else:
            names = [p.stem for p in examples]
            sel = st.selectbox("Choose an example:", names, key="example_select")
            if st.button("Load example", key="load_example_btn"):
                path = next(p for p in examples if p.stem == sel)
                try:
                    st.session_state.config = load_model_from_path(path)
                    st.session_state.validation_report = None
                    st.session_state.solver_report = None
                    _reset_row_editor_state()
                    _goto(0)
                    st.rerun()
                except StorageError as e:
                    st.error(e.message)


# ----------------------------------------------------------------------
# Step 1: Basics
# ----------------------------------------------------------------------

def render_basics(cfg: ModelConfig) -> None:
    st.header("1. Basics")
    cfg.name = st.text_input("Model name", value=cfg.name, key="basics_name")
    cfg.description = st.text_area("Description (optional)",
                                     value=cfg.description, key="basics_desc")
    cfg.dt = st.number_input(
        "How much real time does one step represent? (dt)",
        min_value=1e-9, value=float(cfg.dt), key="basics_dt")
    cfg.n_sessions = int(st.number_input(
        "How many steps should we simulate?", min_value=1,
        max_value=MAX_N_SESSIONS, value=int(cfg.n_sessions), step=1,
        key="basics_n_sessions"))


# ----------------------------------------------------------------------
# Step 2: States
# ----------------------------------------------------------------------

_STATE_FIELDS = [
    _Field("name", "Name", "text", help="e.g. x1 or demand"),
    _Field("label", "Label (optional)", "text"),
    _Field("initial_value", "Starting value", "float"),
    _Field("min", "Minimum (optional)", "optional_float"),
    _Field("max", "Maximum (optional)", "optional_float"),
    _Field("process_noise", "Process noise scale", "float", min_value=0.0,
           help="How much this quantity drifts randomly each step."),
]


def _prune_dynamics(cfg: ModelConfig) -> None:
    valid_names = {s.name for s in cfg.states}
    cfg.dynamics = {k: v for k, v in cfg.dynamics.items() if k in valid_names}


def render_states(cfg: ModelConfig) -> None:
    st.header("2. States")
    st.write("How many quantities does your system keep track of over time?")

    idx = _select_row(cfg.states, "states")
    _row_form(cfg, cfg.states, StateSpec, _STATE_FIELDS, "states", idx,
              on_change=_prune_dynamics)


# ----------------------------------------------------------------------
# Step 3: Actions
# ----------------------------------------------------------------------

_ACTION_FIELDS = [
    _Field("name", "Name", "text", help="e.g. u1 or price"),
    _Field("label", "Label (optional)", "text"),
    _Field("min", "Minimum allowed value (required)", "float"),
    _Field("max", "Maximum allowed value (required)", "float"),
]


def render_actions(cfg: ModelConfig) -> None:
    st.header("3. Actions")
    st.write("How many things can you choose or control at each step?")

    idx = _select_row(cfg.actions, "actions")
    _row_form(cfg, cfg.actions, ActionSpec, _ACTION_FIELDS, "actions", idx)


# ----------------------------------------------------------------------
# Step 4: Unknown parameters
# ----------------------------------------------------------------------

_PARAMETER_FIELDS = [
    _Field("name", "Name", "text"),
    _Field("label", "Label (optional)", "text"),
    _Field("prior_guess", "Your best initial guess", "float"),
    _Field("prior_variance", "How uncertain is that guess?", "float",
           min_value=1e-9,
           help="A larger number means you're less sure -- the "
                "controller will explore more to pin this down."),
    _Field("process_noise", "Drift/process noise scale", "float", min_value=0.0),
    _Field("true_value", "True value (optional -- simulation/testing only)",
           "optional_float",
           help="Only fill this in if you're testing in simulation, "
                "not on real data -- used to check whether the "
                "controller learns correctly."),
]


def render_parameters(cfg: ModelConfig) -> None:
    st.header("4. Unknown parameters")
    st.write(
        "Is there anything in your system whose true value you don't "
        "know, but want the controller to learn from data as it runs? "
        "It's fine to have none.")
    if cfg.control_method == "ilqr" and cfg.parameters:
        st.info(
            "Control method is currently iLQR (Advanced settings), which "
            "has no online estimation -- any parameter defined here is "
            "used at its fixed 'best initial guess' value instead of "
            "being learned. Switch Control method to iLQG / Dual Control "
            "to have the controller estimate it from data.")

    idx = _select_row(cfg.parameters, "parameters")
    _row_form(cfg, cfg.parameters, ParameterSpec, _PARAMETER_FIELDS,
              "parameters", idx)


# ----------------------------------------------------------------------
# Step 5: Constraints
# ----------------------------------------------------------------------

_CONSTRAINT_KIND_OPTIONS = [
    ("state_action", "h(x, u) >= 0  --  may use states, actions, constants"),
    ("state_only", "h(x) >= 0  --  states and constants only "
                    "(reduced via a Lie derivative + alpha)"),
]

_CONSTRAINT_FIELDS = [
    _Field("name", "Name", "text"),
    _Field("alpha", "Alpha", "float", min_value=1e-9,
           help="Only used by an h(x) >= 0 constraint -- controls how "
                "quickly its reduction term decays (Dastan & Sensinger "
                "2024, eq 18-19). Ignored for an h(x, u) >= 0 constraint."),
    _Field("enabled", "Enabled", "bool"),
]


def render_constraints(cfg: ModelConfig) -> None:
    st.header("5. Constraints")
    st.write(
        "Optionally require h(...) >= 0 to hold throughout the run, "
        "enforced through the existing constrained-DDP extension "
        "(Dastan & Sensinger 2024, see extensions/constraints/) -- no new "
        "constraint math, just a generic way to describe one. It's fine "
        "to have none. Action bounds (step 3) always apply regardless.")
    st.caption(
        "**Only enforced in iLQR control method** (Advanced settings, "
        "step 10) -- iLQG / Dual Control has no hook for this. A "
        "constraint only actually restricts anything if it (or, for an "
        "h(x) constraint, its Lie derivative) depends on exactly one "
        "action -- Test This Model will warn you if a constraint you "
        "wrote turns out not to.")

    if (cfg.control_method != "ilqr" and cfg.constraints
            and any(c.enabled for c in cfg.constraints)):
        st.warning(
            "Control method is currently iLQG / Dual Control -- these "
            "constraint(s) will be rejected at Validate time until you "
            "either switch Control method to iLQR (Advanced settings) or "
            "disable/remove them.", icon="⚠️")

    idx = _select_row(cfg.constraints, "constraints")
    current = cfg.constraints[idx] if idx is not None else None

    kind_key = f"constraints_{idx}_kind"
    kind_values = [k for k, _ in _CONSTRAINT_KIND_OPTIONS]
    kind_labels = dict(_CONSTRAINT_KIND_OPTIONS)
    kind = st.selectbox(
        "Kind", options=kind_values,
        index=kind_values.index(current.kind) if current and current.kind in kind_values else 0,
        format_func=lambda k: kind_labels[k], key=kind_key)

    if kind == "state_action":
        allowed_names = cfg.state_names() + cfg.action_names() + cfg.constant_names()
    else:
        allowed_names = cfg.state_names() + cfg.constant_names()
    st.caption(
        f"Known names for this constraint: {', '.join(allowed_names) or '(none defined yet)'}. "
        f"Write h so that h >= 0 is the condition you want to hold.")

    expr_key = f"constraints_{idx}_expression"
    expression = st.text_input(
        "Expression (h)", value=current.expression if current else "",
        key=expr_key)
    _show_equation_feedback(expression, allowed_names)

    _row_form(cfg, cfg.constraints, ConstraintSpec, _CONSTRAINT_FIELDS,
              "constraints", idx,
              extra_values={"kind": kind, "expression": expression},
              extra_keys_to_clear=[kind_key, expr_key])


# ----------------------------------------------------------------------
# Step 6: Constants
# ----------------------------------------------------------------------

_CONSTANT_FIELDS = [
    _Field("name", "Name", "text"),
    _Field("value", "Value", "float"),
]


def render_constants(cfg: ModelConfig) -> None:
    st.header("6. Constants")
    st.write("Any other fixed, known numbers your equations need?")

    idx = _select_row(cfg.constants, "constants")
    _row_form(cfg, cfg.constants, ConstantSpec, _CONSTANT_FIELDS, "constants", idx)


# ----------------------------------------------------------------------
# Step 7: Dynamics
# ----------------------------------------------------------------------

def render_dynamics(cfg: ModelConfig) -> None:
    st.header("7. Dynamics")
    all_names = cfg.all_defined_names()
    _equation_syntax_caption(all_names)

    if not cfg.states:
        st.warning("Define at least one state first (step 2).")
        return

    for s in cfg.states:
        text = st.text_input(
            f"How does '{s.name}' change over time? (its rate of change)",
            value=cfg.dynamics.get(s.name, ""), key=f"dyn_{s.name}")
        cfg.dynamics[s.name] = text
        _show_equation_feedback(text, all_names)


# ----------------------------------------------------------------------
# Step 8: Measurement
# ----------------------------------------------------------------------

_MEASUREMENT_FIELDS = [
    _Field("name", "Output name", "text"),
    _Field("noise_scale", "Measurement noise scale", "float", min_value=0.0),
]


def render_measurement(cfg: ModelConfig) -> None:
    st.header("8. Measurement")
    st.write("What can actually be observed/measured about your system?")
    all_names = cfg.all_defined_names()
    _equation_syntax_caption(all_names)
    st.caption("Tip: to observe a state directly, just write its name.")

    # `expression` lives OUTSIDE the row-editor form (unlike name/noise_scale
    # below) specifically so it can get live equation-syntax feedback as you
    # type -- st.form only reports its contents on submit, so a field inside
    # one can't have per-keystroke feedback the way Dynamics/Cost do.
    idx = _select_row(cfg.measurement, "measurement")
    current = cfg.measurement[idx] if idx is not None else None
    expr_key = f"measurement_{idx}_expression"
    expression = st.text_input(
        "Expression", value=current.expression if current else "",
        key=expr_key, help="e.g. a state name, to observe it directly")
    _show_equation_feedback(expression, all_names)

    _row_form(cfg, cfg.measurement, MeasurementSpec, _MEASUREMENT_FIELDS,
              "measurement", idx, extra_values={"expression": expression},
              extra_keys_to_clear=[expr_key])


# ----------------------------------------------------------------------
# Step 9: Cost
# ----------------------------------------------------------------------

def render_cost(cfg: ModelConfig) -> None:
    st.header("9. Cost")
    st.write(
        "Write an expression that should be SMALL when things are going "
        "well and LARGE when they're going badly -- this is what the "
        "controller tries to minimize each step.")
    all_names = cfg.all_defined_names()
    _equation_syntax_caption(all_names)

    cfg.cost.running = st.text_area(
        "Running cost (evaluated every step)", value=cfg.cost.running,
        key="cost_running")
    _show_equation_feedback(cfg.cost.running, all_names)

    cfg.cost.terminal = st.text_input(
        "Terminal cost (evaluated only at the final step -- use '0' if "
        "there's no separate terminal cost)",
        value=cfg.cost.terminal, key="cost_terminal")
    _show_equation_feedback(cfg.cost.terminal, all_names)

    st.info(
        "Tip: if your cost only reacts to an action indirectly, through "
        "a state that changes with a lag, the controller may not treat "
        "today's action as having any immediate cost. If your system "
        "responds to an action right away, consider writing the cost in "
        "terms of that immediate effect rather than only the lagged state.")


# ----------------------------------------------------------------------
# Step 10: Advanced settings
# ----------------------------------------------------------------------

_CONTROL_METHOD_LABELS = {
    "ilqg": "iLQG / Dual Control -- estimates unknown parameters online "
             "while controlling (MPC replanning every step)",
    "ilqr": "iLQR -- one deterministic full-horizon solve, no estimation "
             "(required for step 5's constraints)",
}


def render_advanced(cfg: ModelConfig) -> None:
    st.header("10. Advanced settings")

    cfg.control_method = st.radio(
        "Control method", options=list(CONTROL_METHODS),
        index=list(CONTROL_METHODS).index(cfg.control_method)
        if cfg.control_method in CONTROL_METHODS else 0,
        format_func=lambda m: _CONTROL_METHOD_LABELS[m], key="adv_control_method")
    st.divider()

    s = cfg.solver

    method_options = [1, 2]
    s.u_lim_method = st.selectbox(
        "Bound enforcement method", options=method_options,
        index=method_options.index(s.u_lim_method) if s.u_lim_method in method_options else 0,
        format_func=lambda v: ("Box-QP (recommended)" if v == 1
                                 else "Tanh-squash (can trap the controller "
                                      "if the initial guess is very wrong)"),
        key="adv_u_lim_method")

    reg_options = [1, 2, 3]
    s.reg_type = st.selectbox(
        "Regularization type", options=reg_options,
        index=reg_options.index(s.reg_type) if s.reg_type in reg_options else 0,
        key="adv_reg_type")

    s.max_du_iterations = int(st.number_input(
        "Planning iterations per step", min_value=1,
        max_value=MAX_DU_ITERATIONS, value=int(s.max_du_iterations),
        key="adv_max_du"))
    s.first_run_max_du_iterations = int(st.number_input(
        "Planning iterations for the very first step", min_value=1,
        max_value=MAX_FIRST_RUN_DU_ITERATIONS,
        value=int(s.first_run_max_du_iterations), key="adv_first_du"))
    s.seed = int(st.number_input("Random seed", value=int(s.seed), step=1,
                                    key="adv_seed"))


# ----------------------------------------------------------------------
# Step 11: Review & Save
# ----------------------------------------------------------------------

def render_review(cfg: ModelConfig) -> None:
    st.header("11. Review & Save")

    errors = cfg.validate()
    if errors:
        st.warning("Some things still need attention:")
        for e in errors:
            st.write(f"- {e}")
    else:
        st.success("This model looks structurally complete.")

    with st.expander("Full configuration (JSON)"):
        st.json(cfg.to_dict())

    with st.expander("View generated code (preview only -- see note below)"):
        st.caption(
            "This is a READ-ONLY preview of what your model looks like as "
            "Python source, for reference or to copy into a real "
            "applications/ folder later. The wizard itself always runs "
            "your model through its own tested engine, never by executing "
            "this text.")
        for fname, src in generate_all_previews(cfg).items():
            st.caption(fname)
            st.code(src, language="python")

    st.download_button(
        "Download model file (recommended -- survives a server restart)",
        json.dumps(cfg.to_dict(), indent=2),
        file_name=f"{cfg.name or 'model'}.json", mime="application/json",
        key="download_model_btn")

    if st.button("Save model to this container (temporary, see warning above)",
                 key="save_btn"):
        try:
            path = save_model(cfg)
            st.success(f"Saved to {path.name}")
        except StorageError as e:
            st.error(e.message)


# ----------------------------------------------------------------------
# Step 12: Validate
# ----------------------------------------------------------------------

def render_validate(cfg: ModelConfig) -> None:
    st.header("12. Validate")
    st.write(
        "This checks your equations and settings against sample numbers "
        "before anything touches the real solver.")

    if st.button("Test This Model", key="validate_btn"):
        with st.status("Testing your model...", expanded=True) as status:
            report = run_validation(cfg)
            st.session_state.validation_report = report
            st.session_state.validated_config_signature = _config_signature(cfg)
            status.update(label="Done", state="complete" if report.ok else "error")

    report = st.session_state.validation_report
    if report is not None:
        stale = st.session_state.validated_config_signature != _config_signature(cfg)
        if stale:
            st.info("The model has changed since this result -- test again.")
        if report.ok:
            st.success("Model passed all checks. You can proceed to Run & Results.")
            for w in report.warnings:
                st.warning(w, icon="⚠️")
        else:
            st.error("Problems found:")
            for m in report.messages:
                st.write(f"- {m}")
            if report.raw_error:
                with st.expander("Technical details"):
                    st.code(report.raw_error)


# ----------------------------------------------------------------------
# Step 13: Run & Results
# ----------------------------------------------------------------------

def render_run(cfg: ModelConfig) -> None:
    st.header("13. Run & Results")

    sig = _config_signature(cfg)
    validated_ok = (st.session_state.validation_report is not None
                     and st.session_state.validation_report.ok
                     and st.session_state.validated_config_signature == sig)
    if not validated_ok:
        st.warning(
            "Please pass validation (step 12) first -- and re-validate if "
            "you changed anything since.")

    if st.button("Run Simulation", key="run_btn", disabled=not validated_ok):
        with st.status(
                f"Running the solver (up to {cfg.n_sessions} steps, "
                f"{cfg.solver.max_du_iterations} planning iterations each -- "
                f"this may take a little while)...", expanded=True) as status:
            report = run_solver(cfg)
            st.session_state.solver_report = report
            if report.ok:
                status.update(label=f"Done in {report.elapsed_seconds:.1f}s",
                                state="complete")
            else:
                status.update(label="Run did not complete", state="error")

    report = st.session_state.solver_report
    if report is None:
        return

    if not report.ok:
        st.error("The run did not complete:")
        for m in report.messages:
            st.write(f"- {m}")
        if report.raw_error:
            with st.expander("Technical details"):
                st.code(report.raw_error)
        return

    result = report.result
    st.success(
        f"Completed in {report.elapsed_seconds:.1f}s. "
        f"Total cost over the run: {result['total_true_cost']:.4g}")

    if "stop_reason" in result:  # iLQR only -- iLQG's MPC loop has no single "converged" moment
        stop_reason = result["stop_reason"]
        if isinstance(stop_reason, str) and stop_reason.startswith("SUCCESS"):
            st.info(f"Optimizer: {stop_reason}")
        else:
            st.warning(
                f"Optimizer: {stop_reason}. This does not necessarily mean "
                f"the result below is unusable -- check the trajectory and "
                f"convergence plots; a constrained problem can make real "
                f"progress and still end this way (see the 'EXIT: lambda > "
                f"lambda_max' discussion in applications/"
                f"pendulum_constrained/run_zahid_demo.py).")

    fig = plot_convergence(result)
    if fig is not None:
        st.pyplot(fig)
    fig = plot_states(result, cfg)
    if fig is not None:
        st.pyplot(fig)
    fig = plot_actions(result, cfg)
    if fig is not None:
        st.pyplot(fig)
    fig = plot_parameters(result, cfg)
    if fig is not None:
        st.pyplot(fig)
    fig = plot_cost(result)
    if fig is not None:
        st.pyplot(fig)

    df = results_table(result, cfg)
    if df is not None:
        st.subheader("Results table")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download results as CSV", df.to_csv(index=False),
            file_name=f"{cfg.name}_results.csv", key="download_csv")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

RENDERERS = [
    render_basics, render_states, render_actions, render_parameters,
    render_constraints, render_constants, render_dynamics, render_measurement,
    render_cost, render_advanced, render_review, render_validate, render_run,
]


def main() -> None:
    st.set_page_config(page_title="Model Wizard", layout="wide")
    _init_state()

    # "Start over" (below) can't call _goto(0) directly: by the time its
    # button is clicked, the sidebar's step_idx radio has ALREADY been
    # instantiated earlier in this same script run (see _goto's own
    # docstring), and Streamlit refuses to overwrite a widget's
    # session_state value after that widget has rendered this run. Defer
    # the actual reset to the TOP of the next run, before the radio exists.
    if st.session_state.pop("_pending_step_reset", False):
        _goto(0)

    cfg = st.session_state.config

    with st.sidebar:
        st.title("Model Wizard")
        if cfg is not None:
            st.caption(f"Editing: **{cfg.name or '(unnamed)'}**")
            st.radio("Steps", options=list(range(len(STEP_NAMES))),
                       format_func=lambda i: STEP_NAMES[i], key="step_idx")
            st.divider()
            st.download_button(
                "Download model file", json.dumps(cfg.to_dict(), indent=2),
                file_name=f"{cfg.name or 'model'}.json", mime="application/json",
                key="sidebar_download")
            if st.button("Save now (temporary, this container only)",
                         key="sidebar_save"):
                try:
                    path = save_model(cfg)
                    st.success(f"Saved to {path.name}")
                except StorageError as e:
                    st.error(e.message)
            if st.button("Start over", key="sidebar_reset"):
                st.session_state.config = None
                st.session_state.validation_report = None
                st.session_state.solver_report = None
                _reset_row_editor_state()
                st.session_state["_pending_step_reset"] = True
                st.rerun()

    if cfg is None:
        render_start()
        return

    idx = max(0, min(st.session_state.step_idx, len(RENDERERS) - 1))
    try:
        RENDERERS[idx](cfg)
    except Exception as e:  # noqa: BLE001 -- absolute UI-level backstop
        st.error(
            "Something unexpected went wrong displaying this step. Your "
            "model's data hasn't been lost -- try 'Save now' in the "
            "sidebar, then go back and adjust your last change.")
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
