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

import json

import numpy as np
import pandas as pd
import streamlit as st

from wizard.codegen_preview import generate_all_previews
from wizard.equation_parser import ALLOWED_FUNCTIONS, check_equation_live
from wizard.plotting import (
    plot_actions, plot_cost, plot_parameters, plot_states, results_table,
)
from wizard.schema import (
    MAX_DU_ITERATIONS, MAX_FIRST_RUN_DU_ITERATIONS, MAX_N_SESSIONS,
    ActionSpec, ConstantSpec, MeasurementSpec, ModelConfig, ParameterSpec,
    StateSpec,
)
from wizard.solver_runner import run_solver
from wizard.storage import (
    StorageError, list_examples, list_models, load_model, load_model_from_path,
    save_model,
)
from wizard.validation_runner import run_validation

STEP_NAMES = [
    "1. Basics", "2. States", "3. Actions", "4. Unknown parameters",
    "5. Constants", "6. Dynamics", "7. Measurement", "8. Cost",
    "9. Advanced settings", "10. Review & Save", "11. Validate",
    "12. Run & Results",
]


def _init_state() -> None:
    st.session_state.setdefault("config", None)
    st.session_state.setdefault("step_idx", 0)
    st.session_state.setdefault("validation_report", None)
    st.session_state.setdefault("validated_config_signature", None)
    st.session_state.setdefault("solver_report", None)


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


def _nullable_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _non_empty_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _show_equation_feedback(text: str, all_names: list[str]) -> None:
    if not text.strip():
        return
    ok, msg = check_equation_live(text, all_names)
    if ok:
        st.success("Looks good.", icon="✅")
    else:
        st.error(msg, icon="⚠️")


# ----------------------------------------------------------------------
# Start screen (shown when no model is loaded/created yet)
# ----------------------------------------------------------------------

def render_start() -> None:
    st.title("Model Wizard")
    st.write(
        "Build and test a control/estimation model through guided "
        "questions -- states, actions, unknown parameters, dynamics, "
        "measurement, and cost -- with no code required.")

    choice = st.radio(
        "What would you like to do?",
        ["Start a new model", "Load a saved model", "Load an example"],
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
                _goto(0)
                st.rerun()

    elif choice == "Load a saved model":
        names = list_models()
        if not names:
            st.info("No saved models yet.")
        else:
            sel = st.selectbox("Choose a saved model:", names, key="load_select")
            if st.button("Load", key="load_btn"):
                try:
                    st.session_state.config = load_model(sel)
                    st.session_state.validation_report = None
                    st.session_state.solver_report = None
                    _goto(0)
                    st.rerun()
                except StorageError as e:
                    st.error(e.message)

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

def render_states(cfg: ModelConfig) -> None:
    st.header("2. States")
    st.write("How many quantities does your system keep track of over time?")

    df = pd.DataFrame({
        "name": pd.Series([s.name for s in cfg.states], dtype="object"),
        "label": pd.Series([s.label for s in cfg.states], dtype="object"),
        "initial_value": pd.Series([s.initial_value for s in cfg.states], dtype="float64"),
        "min": pd.Series([s.min for s in cfg.states], dtype="float64"),
        "max": pd.Series([s.max for s in cfg.states], dtype="float64"),
        "process_noise": pd.Series([s.process_noise for s in cfg.states], dtype="float64"),
    })
    edited = st.data_editor(
        df, num_rows="dynamic", key="states_editor", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name", help="e.g. x1 or demand"),
            "label": st.column_config.TextColumn("Label (optional)"),
            "initial_value": st.column_config.NumberColumn("Starting value"),
            "min": st.column_config.NumberColumn("Minimum (optional)"),
            "max": st.column_config.NumberColumn("Maximum (optional)"),
            "process_noise": st.column_config.NumberColumn(
                "Process noise scale", min_value=0.0,
                help="How much this quantity drifts randomly each step."),
        })

    new_states = []
    for _, row in edited.iterrows():
        name = _non_empty_str(row.get("name")).strip()
        if not name:
            continue
        new_states.append(StateSpec(
            name=name, label=_non_empty_str(row.get("label")),
            initial_value=_nullable_float(row.get("initial_value")) or 0.0,
            min=_nullable_float(row.get("min")),
            max=_nullable_float(row.get("max")),
            process_noise=_nullable_float(row.get("process_noise")) or 1.0))
    cfg.states = new_states

    valid_names = {s.name for s in cfg.states}
    cfg.dynamics = {k: v for k, v in cfg.dynamics.items() if k in valid_names}


# ----------------------------------------------------------------------
# Step 3: Actions
# ----------------------------------------------------------------------

def render_actions(cfg: ModelConfig) -> None:
    st.header("3. Actions")
    st.write("How many things can you choose or control at each step?")

    df = pd.DataFrame({
        "name": pd.Series([a.name for a in cfg.actions], dtype="object"),
        "label": pd.Series([a.label for a in cfg.actions], dtype="object"),
        "min": pd.Series([a.min for a in cfg.actions], dtype="float64"),
        "max": pd.Series([a.max for a in cfg.actions], dtype="float64"),
    })
    edited = st.data_editor(
        df, num_rows="dynamic", key="actions_editor", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name", help="e.g. u1 or price"),
            "label": st.column_config.TextColumn("Label (optional)"),
            "min": st.column_config.NumberColumn("Minimum allowed value (required)"),
            "max": st.column_config.NumberColumn("Maximum allowed value (required)"),
        })

    new_actions = []
    for _, row in edited.iterrows():
        name = _non_empty_str(row.get("name")).strip()
        if not name:
            continue
        lo = _nullable_float(row.get("min"))
        hi = _nullable_float(row.get("max"))
        new_actions.append(ActionSpec(
            name=name, label=_non_empty_str(row.get("label")),
            min=lo if lo is not None else 0.0,
            max=hi if hi is not None else 1.0))
    cfg.actions = new_actions


# ----------------------------------------------------------------------
# Step 4: Unknown parameters
# ----------------------------------------------------------------------

def render_parameters(cfg: ModelConfig) -> None:
    st.header("4. Unknown parameters")
    st.write(
        "Is there anything in your system whose true value you don't "
        "know, but want the controller to learn from data as it runs? "
        "It's fine to have none.")

    df = pd.DataFrame({
        "name": pd.Series([p.name for p in cfg.parameters], dtype="object"),
        "label": pd.Series([p.label for p in cfg.parameters], dtype="object"),
        "prior_guess": pd.Series([p.prior_guess for p in cfg.parameters], dtype="float64"),
        "prior_variance": pd.Series([p.prior_variance for p in cfg.parameters], dtype="float64"),
        "process_noise": pd.Series([p.process_noise for p in cfg.parameters], dtype="float64"),
        "true_value": pd.Series([p.true_value for p in cfg.parameters], dtype="float64"),
    })
    edited = st.data_editor(
        df, num_rows="dynamic", key="parameters_editor", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "label": st.column_config.TextColumn("Label (optional)"),
            "prior_guess": st.column_config.NumberColumn("Your best initial guess"),
            "prior_variance": st.column_config.NumberColumn(
                "How uncertain is that guess?", min_value=1e-9,
                help="A larger number means you're less sure -- the "
                     "controller will explore more to pin this down."),
            "process_noise": st.column_config.NumberColumn(
                "Drift/process noise scale", min_value=0.0),
            "true_value": st.column_config.NumberColumn(
                "True value (optional -- simulation/testing only)",
                help="Only fill this in if you're testing in simulation, "
                     "not on real data -- used to check whether the "
                     "controller learns correctly."),
        })

    new_params = []
    for _, row in edited.iterrows():
        name = _non_empty_str(row.get("name")).strip()
        if not name:
            continue
        new_params.append(ParameterSpec(
            name=name, label=_non_empty_str(row.get("label")),
            prior_guess=_nullable_float(row.get("prior_guess")) or 0.0,
            prior_variance=_nullable_float(row.get("prior_variance")) or 1.0,
            process_noise=_nullable_float(row.get("process_noise")) or 0.05,
            true_value=_nullable_float(row.get("true_value"))))
    cfg.parameters = new_params


# ----------------------------------------------------------------------
# Step 5: Constants
# ----------------------------------------------------------------------

def render_constants(cfg: ModelConfig) -> None:
    st.header("5. Constants")
    st.write("Any other fixed, known numbers your equations need?")

    df = pd.DataFrame({
        "name": pd.Series([c.name for c in cfg.constants], dtype="object"),
        "value": pd.Series([c.value for c in cfg.constants], dtype="float64"),
    })
    edited = st.data_editor(
        df, num_rows="dynamic", key="constants_editor", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "value": st.column_config.NumberColumn("Value"),
        })

    new_constants = []
    for _, row in edited.iterrows():
        name = _non_empty_str(row.get("name")).strip()
        if not name:
            continue
        new_constants.append(ConstantSpec(
            name=name, value=_nullable_float(row.get("value")) or 0.0))
    cfg.constants = new_constants


# ----------------------------------------------------------------------
# Step 6: Dynamics
# ----------------------------------------------------------------------

def render_dynamics(cfg: ModelConfig) -> None:
    st.header("6. Dynamics")
    all_names = cfg.all_defined_names()
    st.caption(
        f"Known names: {', '.join(all_names) or '(none defined yet)'}. "
        f"Allowed functions: {', '.join(sorted(ALLOWED_FUNCTIONS))}. "
        f"Write multiplication explicitly (2*x1, not 2x1).")

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
# Step 7: Measurement
# ----------------------------------------------------------------------

def render_measurement(cfg: ModelConfig) -> None:
    st.header("7. Measurement")
    st.write("What can actually be observed/measured about your system?")
    all_names = cfg.all_defined_names()
    st.caption(
        f"Known names: {', '.join(all_names) or '(none defined yet)'}. "
        f"Tip: to observe a state directly, just write its name.")

    df = pd.DataFrame({
        "name": pd.Series([m.name for m in cfg.measurement], dtype="object"),
        "expression": pd.Series([m.expression for m in cfg.measurement], dtype="object"),
        "noise_scale": pd.Series([m.noise_scale for m in cfg.measurement], dtype="float64"),
    })
    edited = st.data_editor(
        df, num_rows="dynamic", key="measurement_editor", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Output name"),
            "expression": st.column_config.TextColumn(
                "Expression", help="e.g. a state name, to observe it directly"),
            "noise_scale": st.column_config.NumberColumn(
                "Measurement noise scale", min_value=0.0),
        })

    new_measurement = []
    for _, row in edited.iterrows():
        name = _non_empty_str(row.get("name")).strip()
        if not name:
            continue
        expr = _non_empty_str(row.get("expression"))
        new_measurement.append(MeasurementSpec(
            name=name, expression=expr,
            noise_scale=_nullable_float(row.get("noise_scale")) or 1.0))
    cfg.measurement = new_measurement

    for m in cfg.measurement:
        if m.expression.strip():
            ok, msg = check_equation_live(m.expression, all_names)
            if not ok:
                st.error(f"'{m.name}': {msg}", icon="⚠️")


# ----------------------------------------------------------------------
# Step 8: Cost
# ----------------------------------------------------------------------

def render_cost(cfg: ModelConfig) -> None:
    st.header("8. Cost")
    st.write(
        "Write an expression that should be SMALL when things are going "
        "well and LARGE when they're going badly -- this is what the "
        "controller tries to minimize each step.")
    all_names = cfg.all_defined_names()
    st.caption(f"Known names: {', '.join(all_names) or '(none defined yet)'}.")

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
# Step 9: Advanced settings
# ----------------------------------------------------------------------

def render_advanced(cfg: ModelConfig) -> None:
    st.header("9. Advanced settings")
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
# Step 10: Review & Save
# ----------------------------------------------------------------------

def render_review(cfg: ModelConfig) -> None:
    st.header("10. Review & Save")

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

    if st.button("Save model", key="save_btn"):
        try:
            path = save_model(cfg)
            st.success(f"Saved to {path.name}")
        except StorageError as e:
            st.error(e.message)


# ----------------------------------------------------------------------
# Step 11: Validate
# ----------------------------------------------------------------------

def render_validate(cfg: ModelConfig) -> None:
    st.header("11. Validate")
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
        else:
            st.error("Problems found:")
            for m in report.messages:
                st.write(f"- {m}")
            if report.raw_error:
                with st.expander("Technical details"):
                    st.code(report.raw_error)


# ----------------------------------------------------------------------
# Step 12: Run & Results
# ----------------------------------------------------------------------

def render_run(cfg: ModelConfig) -> None:
    st.header("12. Run & Results")

    sig = _config_signature(cfg)
    validated_ok = (st.session_state.validation_report is not None
                     and st.session_state.validation_report.ok
                     and st.session_state.validated_config_signature == sig)
    if not validated_ok:
        st.warning(
            "Please pass validation (step 11) first -- and re-validate if "
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
    render_constants, render_dynamics, render_measurement, render_cost,
    render_advanced, render_review, render_validate, render_run,
]


def main() -> None:
    st.set_page_config(page_title="Model Wizard", layout="wide")
    _init_state()

    cfg = st.session_state.config

    with st.sidebar:
        st.title("Model Wizard")
        if cfg is not None:
            st.caption(f"Editing: **{cfg.name or '(unnamed)'}**")
            st.radio("Steps", options=list(range(len(STEP_NAMES))),
                       format_func=lambda i: STEP_NAMES[i], key="step_idx")
            st.divider()
            if st.button("Save now", key="sidebar_save"):
                try:
                    path = save_model(cfg)
                    st.success(f"Saved to {path.name}")
                except StorageError as e:
                    st.error(e.message)
            if st.button("Start over", key="sidebar_reset"):
                st.session_state.config = None
                st.session_state.validation_report = None
                st.session_state.solver_report = None
                _goto(0)
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
