"""
Generate a clinically grounded, persona-modulated synthetic SCI
rehabilitation dataset, for SINDy pipeline development and validation.

Ported from Katie's CODES/KATIE'S CODE/.../generate_synthetic_dataset.m.

This is the DATA GENERATOR that feeds the whole SINDy pipeline: its X/A
output is exactly what build_sindy_library.py expects
((n_states, n_sessions, n_patients) / (n_actions, n_sessions, n_patients)).

13-THERAPY MODEL: each of the 13 WHO therapies (who_therapies.py)
contributes to the two dynamic states (SCIM, BBS) via a base gain
matrix G, scaled per-persona by therapy_efficacy -- see
define_persona.py's module docstring for why this (not the cost
function) is what makes the optimal therapy MIX differ by persona.

PERSONA MODULATION: cfg['persona'] (a define_persona.py Persona)
modulates:
  - patient attribute sampling (AIS priors, age, DPI, caregiver)
  - per-therapy EFFECTIVE dose: motivation (global gain),
    therapy_efficacy (per-therapy response), pain_sensitivity &
    fatigue_tol (attenuate HIGH-LOAD therapies), comorbidity
    (recovery-ceiling penalty)

STATE VARIABLES (n_states=6): x1 SCIM, x2 BBS (both [0,1], dynamic),
x3 AIS grade in {0.25,0.5,0.75,1.0}, x4 Age (norm.), x5 Days-post-injury
(norm.), x6 Caregiver support -- all fixed, matching
continuous_state_dynamics.py's state layout exactly.

TRUE GOVERNING DYNAMICS (persona-modulated), for each therapy j=1..13:
    u_eff_j = u_j * eff_j * motivation * load_atten_j
    dx1/dt = (-0.05*x1 + 0.30*x1*x3)*head1 + (G_scim . u_eff)*(1+0.30*x3+0.15*x6)*soft1*0.20
    dx2/dt = (-0.04*x2 + 0.22*x2*x3)*head2 + (G_bbs  . u_eff)*(1+0.15*x6)        *soft2*0.20
    head_i = clip(1 - x_i/ceiling, 0, 1)   -- STRONG saturation on the homeostatic terms
    soft_i = 0.55 + 0.45*head_i            -- MILD, near-flat saturation on the therapy
                                               terms (keeps therapy effects SINDy-
                                               identifiable even near the recovery ceiling)

SPARSE PER-SESSION THERAPY ACTIVATION: only 3-5 of the 13 therapies are
dosed each session (not all 13 at once) -- this decorrelates the 13
dose regressors so each therapy's effect is individually identifiable
by SINDy from one-step finite differences; with all 13 active every
session simultaneously, their effects would be collinear and
unrecoverable. A(:,:,p) records the PRESCRIBED (raw) dose -- what a
clinician would set -- not the persona-modulated EFFECTIVE dose that
actually drives the dynamics; SINDy has to discover the persona-
specific response purely from (dose, outcome) pairs, exactly like a
real identification problem.

Xi_true (diagnostic, approximate): calls build_sindy_library.py on a
2-patient slice purely to get the SAME library column/label ordering
the real pipeline will use, then hand-fills the KNOWN linear/base
coefficients at a representative mid-working-point -- this does NOT
capture the saturation/cross-term nonlinearity above, so it's a rough
reference for coefficient-recovery diagnostics, not exact ground truth.

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - randsample/betarnd (MATLAB Statistics Toolbox) replaced with
    numpy.random.Generator.choice/.beta, per the existing "MATLAB-Stats-
    Toolbox shims" decision in CLAUDE.md's Katie conversion plan.
  - Explicit `rng: np.random.Generator` parameter added (default: a
    fresh default_rng()) -- MATLAB's implicit global RNG state has no
    Python equivalent to preserve, same reasoning already applied to
    cross_validate_lambda.py's `random_state` parameter. Needed for
    reproducible tests.
  - `clip(v, lo, hi)`, a trivial MATLAB local helper, is np.clip -- an
    exact NumPy equivalent, not reimplemented.
  - cfg_label's n_patients/n_sessions/noise_std fields (in the MATLAB
    source's throwaway call to build_sindy_library for label ordering)
    are dead: build_sindy_library.m/py reads dimensions from the ARRAY
    SHAPES it's given, never from cfg. Dropped rather than replicated.
  - The 'x3*x2' case in the Xi_true label switch is unreachable given
    build_sindy_library.py's own column-generation order (cross terms
    are always emitted i<j, so state2*state3 always renders as
    "x2*x3", never "x3*x2") -- kept anyway for literal fidelity with
    the MATLAB source's defensive-but-dead branch.
  - COMMENT/CODE MISMATCH FOUND IN THE MATLAB SOURCE (flagged, not
    silently reconciled): the file's top-level docstring summarises the
    saturation as `sat_i = max(0, ceiling_adjusted - x_i)` (additive),
    but the actual simulation code implements
    `head_i = clip(1 - x_i/ceiling, 0, 1)` (a ratio) -- materially
    different formulas. This port implements the REAL code (the ratio
    form), matching what generate_synthetic_dataset.m actually runs.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from extensions.sindy.build_sindy_library import build_sindy_library
from applications.sci_rehab.define_persona import Persona, define_persona
from applications.sci_rehab.who_therapies import who_therapies


@dataclass
class SyntheticDataset:
    X: np.ndarray               # (n_states, n_sessions, n_patients)
    A: np.ndarray                # (n_actions, n_sessions, n_patients) -- PRESCRIBED doses
    Y: np.ndarray                 # (n_outcomes, n_sessions, n_patients)
    Xi_true: np.ndarray            # (n_terms, n_states) approximate diagnostic ground truth
    swat_ceiling: np.ndarray        # (n_patients,) per-patient recovery ceiling
    state_names: List[str]
    action_names: List[str]
    therapy_names: List[str]
    outcome_names: List[str]
    n_states: int
    n_actions: int
    n_outcomes: int
    n_patients: int
    n_sessions: int
    persona: Persona


def generate_synthetic_dataset(cfg, rng: Optional[np.random.Generator] = None) -> SyntheticDataset:
    if rng is None:
        rng = np.random.default_rng()

    n_p = cfg["n_patients"]
    n_s = cfg["n_sessions"]
    noise_std = cfg["noise_std"]

    # -- persona parameters --
    prs = cfg.get("persona") or define_persona("default")

    # -- WHO therapy definition (13 control signals) --
    TH = who_therapies()
    n_actions = TH.n
    G_scim = TH.G[0, :]
    G_bbs = TH.G[1, :]
    load = TH.load
    eff = np.asarray(prs.therapy_efficacy)

    # -- patient attribute sampling (persona-modulated) --
    ais_vals = np.array([0.25, 0.50, 0.75, 1.0])
    ais_idx = rng.choice(4, size=n_p, replace=True, p=prs.ais_probs)
    ais = ais_vals[ais_idx]

    age_lo, age_hi = prs.age_range
    age_raw = np.clip(age_lo + (age_hi - age_lo) * rng.beta(2, 3, size=n_p), age_lo, age_hi)
    age = (age_raw - age_lo) / max(age_hi - age_lo, 1)

    dpi_lo, dpi_hi = prs.dpi_range
    dpi_raw = dpi_lo + (dpi_hi - dpi_lo) * rng.random(n_p)
    dpi = (dpi_raw - dpi_lo) / max(dpi_hi - dpi_lo, 1)

    cg = rng.beta(prs.caregiver_alpha, prs.caregiver_beta, size=n_p)

    scim0 = np.clip(ais * 0.5 + 0.1 * rng.standard_normal(n_p), 0.01, 1.0)
    bbs0 = np.clip(ais * 0.45 + 0.1 * scim0 + 0.08 * rng.standard_normal(n_p), 0.01, 1.0)

    # comorbidity reduces the achievable functional ceiling
    swat_ceiling = (0.4 * ais + 0.2 * (1 - age) + 0.2 * cg + 0.1 * (1 - dpi)
                      + 0.1 * rng.random(n_p))
    swat_ceiling = swat_ceiling * (1 - 0.5 * prs.comorbidity)  # persona ceiling penalty
    swat_ceiling = np.clip(swat_ceiling, 0.1, 1.0)

    # -- simulate trajectories --
    n_states = 6
    n_outcomes = 1

    X = np.zeros((n_states, n_s, n_p))
    A = np.zeros((n_actions, n_s, n_p))
    Y = np.zeros((n_outcomes, n_s, n_p))

    for p in range(n_p):
        x3, x4, x5, x6 = ais[p], age[p], dpi[p], cg[p]
        ceil_p = swat_ceiling[p]

        x = np.array([scim0[p], bbs0[p], x3, x4, x5, x6])
        X[:, 0, p] = x + noise_std * rng.standard_normal(n_states)

        for s in range(n_s - 1):
            # SPARSE per-session therapy activation -- see module docstring.
            n_active = rng.integers(3, 6)  # 3-5 therapies this session
            active = rng.permutation(n_actions)[:n_active]
            u = np.zeros(n_actions)
            u[active] = np.clip(0.40 + 0.60 * rng.random(n_active), 0, 1)

            # persona-modulated per-therapy effective dose
            pain_atten = 1 - prs.pain_sensitivity * (load * np.maximum(0, u - 0.60))
            fatigue_atten = 1 - (1 - prs.fatigue_tol) * 0.5 * load
            load_atten = np.maximum(0, pain_atten * fatigue_atten)

            u_eff = u * eff * prs.motivation * load_atten
            u_eff = np.clip(u_eff, 0, 1.5)

            A[:, s, p] = u  # store the PRESCRIBED doses, not u_eff

            x1, x2 = x[0], x[1]

            head1 = np.clip(1 - x1 / max(ceil_p, 0.05), 0, 1)
            head2 = np.clip(1 - x2 / max(ceil_p, 0.05), 0, 1)

            contrib_scim = (G_scim @ u_eff) * (1 + 0.30 * x3 + 0.15 * x6)
            contrib_bbs = (G_bbs @ u_eff) * (1 + 0.15 * x6)

            soft1 = 0.55 + 0.45 * head1
            soft2 = 0.55 + 0.45 * head2

            dx1 = (-0.05 * x1 + 0.30 * x1 * x3) * head1 + contrib_scim * soft1 * 0.20
            dx2 = (-0.04 * x2 + 0.22 * x2 * x3) * head2 + contrib_bbs * soft2 * 0.20

            x[0] = np.clip(x[0] + dx1, 0, 1)
            x[1] = np.clip(x[1] + dx2, 0, 1)

            X[:, s + 1, p] = np.clip(x + noise_std * rng.standard_normal(n_states), 0, 1)

            Y[0, s, p] = (dx1 + dx2) / 2 + noise_std * rng.standard_normal()

        A[:, n_s - 1, p] = A[:, n_s - 2, p]  # repeat last session's dose for the terminal column

    # -- Xi_true (approximate ground truth, diagnostic) --
    lib_cfg = {"poly_order": 2, "include_cross": True, "include_trig": False}
    _, _, lib_labels_tmp, _ = build_sindy_library(X[:, :, :2], A[:, :, :2], lib_cfg)
    n_terms = len(lib_labels_tmp)

    Xi_true = np.zeros((n_terms, n_states))
    clean_labels = [_strip_scale_tag(lbl) for lbl in lib_labels_tmp]

    for ti, lbl in enumerate(clean_labels):
        if lbl == "x1":
            Xi_true[ti, 0] = -0.05
        elif lbl == "x1*x3":
            Xi_true[ti, 0] = 0.30
        elif lbl == "x2":
            Xi_true[ti, 1] = -0.04
        elif lbl in ("x2*x3", "x3*x2"):
            Xi_true[ti, 1] = 0.22

        # linear therapy terms a1..a13: representative coefficient at a
        # mid working point (soft ~= 0.8) -- see module docstring
        m = re.fullmatch(r"a(\d+)", lbl)
        if m:
            j = int(m.group(1)) - 1  # 1-indexed WHO code -> 0-indexed array
            if 0 <= j < n_actions:
                base = eff[j] * prs.motivation * 0.20 * 0.80
                Xi_true[ti, 0] = G_scim[j] * base
                Xi_true[ti, 1] = G_bbs[j] * base

    Xi_true[:, 2:6] = 0.0  # static states have no dynamics

    return SyntheticDataset(
        X=X, A=A, Y=Y, Xi_true=Xi_true, swat_ceiling=swat_ceiling,
        state_names=["SCIM_norm", "BBS_norm", "AIS", "Age_norm", "DPI_norm", "Caregiver"],
        action_names=list(TH.codes), therapy_names=list(TH.names),
        outcome_names=["FunctionalGain"],
        n_states=n_states, n_actions=n_actions, n_outcomes=n_outcomes,
        n_patients=n_p, n_sessions=n_s, persona=prs)


def _strip_scale_tag(label):
    return re.sub(r"\s*\[s=[\d.]+\]", "", label).strip()
