"""
Streamlit GUI for the sci_rehab therapy-planning pipeline -- lets
non-technical staff (nurses) pick a patient persona and run the
SINDy-identification + dual-iLQG therapy-planning pipeline without
touching code.

Pure orchestration/presentation layer: every numerical step is an
unmodified call into run_katie_pipeline.py's already-tested functions
(run_identification, select_uncertain_terms, run_dual_control_planning).
No pipeline/solver code is changed or duplicated here.

Uses FULL-SCALE identification settings (run_katie_pipeline.py's own
defaults: n_patients=60/n_sessions=12), not run_katie_pipeline_preview.py's
n_patients=12/n_sessions=6. The preview scale was tried first for a
faster click-and-wait experience, but investigation (2026-08-20) showed
it identifies noisy, often mixed-sign per-therapy linear coefficients
from only 12 synthetic patients -- e.g. several of a1..a13's coefficients
come out negative -- which traps the dual-iLQG planner in a near-all-zero
-dose local optimum for EVERY persona/patient regardless of how much
therapy they actually need (session-by-session plan reads as all ~0).
Full-scale identification (60 patients) gives clean, physically
sensible, nearly-all-positive linear coefficients and the SAME planner
(zero warm start, unmodified w_resource) then produces real, patient-
need-proportional plans (near-zero doses ONLY for patients already close
to their recovery ceiling -- a correct answer, not a bug; substantial,
sometimes fully-saturated doses for patients with a large gap to close).
Warm-starting the optimizer away from zero and/or scaling down
w_resource (both tried) only shift cost/dose slightly on top of an
already-correct full-scale-identified plan -- neither is needed once
identification itself is clean, so neither is used here. Identification
now takes roughly 35-230s per persona (varies with how many active terms
CV/STLSQ settle on) instead of a few seconds; still cached per (persona,
session count, seed) via st.cache_data so revisiting the same persona
doesn't re-run it.

Run with: streamlit run streamlit_app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from applications.sci_rehab.define_persona import define_persona
from applications.sci_rehab.run_katie_pipeline import (
    run_dual_control_planning,
    run_identification,
    select_uncertain_terms,
)
from applications.sci_rehab.who_therapies import who_therapies

# Full-scale identification settings (matches run_katie_pipeline.py's own
# defaults) -- see module docstring for why the preview scale (12
# patients) was dropped: it produced noisy, mixed-sign coefficients that
# trapped the planner near zero dose regardless of patient need.
IDENT_N_PATIENTS = 60
IDENT_N_SESSIONS = 12
IDENT_NOISE_STD = 0.03
SEED = 0

PERSONA_OPTIONS = {
    "default": "Default (population average)",
    "high_motivator": "High Motivator",
    "frail_elder": "Frail Elder",
    "young_athlete": "Young Athlete",
    "low_support": "Low Social Support",
    "pain_sensitive": "Pain Sensitive",
    "p1_maya": "Maya (P1)",
    "sophia": "Sophia",
    "p3_riley": "Riley (P3)",
    "jean_guy": "Jean Guy",
    "p5_daniel": "Daniel (P5)",
    "joan": "Joan",
    "kenny": "Kenny",
    "p8_marcus": "Marcus (P8)",
    "p9_aaliyah": "Aaliyah (P9)",
    "p10_pierre": "Pierre (P10)",
    "p11_niran": "Niran (P11)",
    "p12_lucia": "Lucia (P12)",
}


@st.cache_data(show_spinner=False, persist="disk")
def run_identification_cached(persona_key: str):
    """Runs identification alone, cached on persona_key ONLY (not
    n_sessions -- identification doesn't depend on the planning horizon)
    and persisted to disk so a pre-warmed cache survives across separate
    `streamlit run` processes, not just reruns within one session.

    Split out from run_pipeline (2026-08-20) so that changing the
    session-count slider for an already-identified persona only re-runs
    the cheap planning step, not the expensive (35-230s) identification
    step -- previously the whole pipeline was one cache entry keyed on
    (persona_key, n_sessions), so any slider change discarded the
    already-computed identification and re-ran it from scratch.

    DEVIATION: identification and planning used to share one continuing
    `np.random.default_rng(SEED)` stream (planning's random draws
    depended on exactly how many draws identification had consumed).
    Splitting the cache means planning can no longer resume identification's
    exact rng state, so each step now seeds its own fresh
    np.random.default_rng(SEED) instead -- same seed, decoupled streams.
    This changes exactly which pseudorandom synthetic patient/dataset
    gets drawn versus before, but there was never a correctness
    requirement on which specific draw was used, only that it be
    reproducible -- which it still is."""
    rng = np.random.default_rng(SEED)
    return run_identification(
        persona_key, n_patients=IDENT_N_PATIENTS, n_sessions=IDENT_N_SESSIONS,
        noise_std=IDENT_NOISE_STD, rng=rng, verbose=False)


@st.cache_data(show_spinner=False, persist="disk")
def run_pipeline(persona_key: str, n_sessions: int):
    """Runs dual-control planning for one persona/horizon, reusing the
    persona-only identification cache above. Cached on
    (persona_key, n_sessions) too, since planning itself isn't free."""
    ident = run_identification_cached(persona_key)

    xi_idx_pairs = select_uncertain_terms(ident.Xi, ident.protected_idx, top_k=1)

    rng = np.random.default_rng(SEED)
    result, p_true, swat = run_dual_control_planning(
        ident, xi_idx_pairs, n_sessions=n_sessions, max_du_iterations=20,
        first_run_max_du_iterations=40, rng=rng, verbose=False)

    return ident, xi_idx_pairs, result, p_true, swat


def dose_table(result, n_sessions: int) -> pd.DataFrame:
    """Per-therapy, per-session dose table: (n_sessions-1) x 13,
    doses in [0, 1]. result["u"] is raw control signal; the box
    constraint maps it to a dose via 0.5*tanh(u)+0.5 (same formula
    print_control_report uses for its mean-dose column)."""
    therapies = who_therapies()
    u = result["u"]  # (13, n_sessions-1)
    doses = 0.5 * np.tanh(u) + 0.5
    return pd.DataFrame(
        doses.T, columns=therapies.names,
        index=[f"Session {k}" for k in range(u.shape[1])])


def trajectory_chart(result, swat: float):
    x_true = result["x_true"]
    sessions = np.arange(x_true.shape[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sessions, x_true[0, :], marker="o", label="SCIM")
    ax.plot(sessions, x_true[1, :], marker="o", label="BBS")
    ax.axhline(swat, color="gray", linestyle="--", linewidth=1,
               label=f"Recovery ceiling ({swat:.2f})")
    ax.set_xlabel("Session")
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Predicted SCIM / BBS recovery trajectory")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="SCI Rehab Therapy Planner", layout="wide")
st.title("SCI Rehab Therapy Planner")

with st.sidebar:
    st.header("Patient")
    persona_key = st.selectbox(
        "Persona", options=list(PERSONA_OPTIONS.keys()),
        format_func=lambda k: PERSONA_OPTIONS[k])
    persona = define_persona(persona_key)
    st.caption(persona.description)

    n_sessions = st.slider("Number of therapy sessions to plan", 4, 12, 8)

    run_clicked = st.button("Run Therapy Plan", type="primary")

if run_clicked:
    st.session_state["last_run"] = (persona_key, n_sessions)

if "last_run" in st.session_state:
    run_persona_key, run_n_sessions = st.session_state["last_run"]
    with st.spinner(
            f"Planning therapy for {PERSONA_OPTIONS[run_persona_key]}... "
            "identifying this persona's model can take a couple of minutes "
            "the first time (cached afterward)"):
        ident, xi_idx_pairs, result, p_true, swat = run_pipeline(
            run_persona_key, run_n_sessions)

    st.subheader("Predicted recovery trajectory")
    st.pyplot(trajectory_chart(result, swat))

    st.subheader("Session-by-session therapy plan")
    doses = dose_table(result, run_n_sessions)
    top3 = doses.mean(axis=0).sort_values(ascending=False).head(3)
    st.markdown(
        "**Top recommended therapies (by mean dose):** "
        + ", ".join(f"{name} ({dose:.2f})" for name, dose in top3.items()))
    st.dataframe(doses.style.format("{:.2f}").background_gradient(
        cmap="Blues", vmin=0, vmax=1), width="stretch")

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Identification NMAE", f"{ident.nmae:.3f}",
                help="Validation error of the identified model; accepted if < 1.0")
    col2.metric("Recovery ceiling (swat)", f"{swat:.3f}")
    col3.metric("Total planning cost", f"{result['total_true_cost']:.2f}")

    if xi_idx_pairs:
        with st.expander("Dual-estimated coefficients (identified -> refined vs. true)"):
            rows = []
            for i, (term_idx, state_idx) in enumerate(xi_idx_pairs):
                label = ident.lib_labels[term_idx].split(" [s=")[0]
                state_name = ident.dataset.state_names[state_idx]
                rows.append({
                    "term": f"{label} (d{state_name}/dt)",
                    "identified": result["p_hat"][i, 0],
                    "final estimate": result["p_hat"][i, -1],
                    "true": p_true[i],
                })
            st.dataframe(pd.DataFrame(rows), width="stretch")
else:
    st.info("Select a persona in the sidebar and click **Run Therapy Plan**.")
