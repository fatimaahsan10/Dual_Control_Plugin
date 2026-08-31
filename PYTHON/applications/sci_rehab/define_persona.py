"""
Persona definitions: clinical/demographic priors, behavioural modifiers,
per-therapy efficacy profiles, and cost-function weights that make the
curiosity-driven optimiser recommend a DIFFERENT therapy mix for each
patient type.

Ported from Katie's CODES/KATIE'S CODE/.../define_persona.m.

A persona is either a POPULATION ARCHETYPE (high_motivator, frail_elder,
young_athlete, low_support, pain_sensitive -- ais_probs spread across
grades, representing a type of patient) or one of 12 named SGBA+
(Sex- and Gender-Based Analysis Plus) intersectional personas
(p1_maya, sophia, ..., p12_lucia) -- each of THOSE represents ONE
concrete named/coded patient, not a population, so their ais_probs are
~94% concentrated on that patient's actual AIS grade rather than spread
across a clinical population. See Persona_development.xlsx (not
ported -- a spreadsheet, not code) for the full 45-attribute source
profiles this file's fields were manually derived from.

THERAPY EFFECTIVENESS PER PERSONA: who_therapies.py defines
population-average "true" functional gains per WHO therapy (G[0,:] for
SCIM, G[1,:] for BBS). `therapy_efficacy` here (13 entries, ordered
U1..U13 to match who_therapies.py's column order) MULTIPLIES those base
gains per persona: 1.0 = responds like the population average, >1 =
unusually good responder to that specific therapy, <1 = poor responder.
This product is what generate_synthetic_dataset.py will use (once
ported) to generate each persona's own ground-truth dynamics -- the
persona-specific behaviour is baked into the SIMULATED PHYSIOLOGY
itself, not the cost function, so the controller genuinely discovers
each persona's best therapy mix through the same general SINDy +
iLQG machinery, rather than needing hand-coded per-persona rules.

COST WEIGHTS: w_scim/w_bbs/w_resource/w_terminal_scim/w_terminal_bbs
map directly onto sindy_model_constants.py's PersonaWeights (already
consumed by l_cost.py) -- see persona_weights() below.

Mapping rules used to derive fields from the source SGBA+ profiles
(documented in the MATLAB source so the choices are auditable /
clinician-reviewable, reproduced here verbatim):
  ais_probs        90%+ mass on the persona's stated AIS grade (it is
                    one named patient, not a sampled population)
  age_range        tight band (+/- ~4-5 yrs) around age at evaluation
  dpi_range         tight band reflecting care_continuum (community vs.
                    tertiary/inpatient -> shorter DPI window)
  caregiver_alpha/  derived from caregiver_inhome + family_supports +
    beta            social_supports (strong in-home -> high alpha/low
                    beta; none/institutional-only -> low alpha/high beta)
  motivation        from pre-injury activity level, social capital,
                    employment engagement, and any mental-health
                    comorbidity (e.g. depression lowers motivation)
  fatigue_tol       from age, cardiometabolic/respiratory comorbidities,
                    and post-injury activity level
  pain_sensitivity  from pain-management modality/intensity and
                    comorbidities associated with chronic pain
  comorbidity       from count/severity of comorbidities + other injuries
                    sustained (e.g. concurrent TBI)
  therapy_efficacy  from AIS grade + neuro level (residual motor/sensory
                    function -> which WHO therapies are biomechanically
                    relevant) and comorbidity profile (e.g. cardiac
                    conditions de-prioritise high-load NMES/fitness)
  w_resource        from distance to specialised centre, income, and
                    paramedical/provider access (remote, low-income, or
                    access-limited personas get a higher resource-
                    preservation weight, biasing toward durable,
                    lower-visit-burden therapy mixes)
  w_scim / w_bbs    from injury level (cervical -> relatively higher
                    w_scim for ADL/independence) and any persona-specific
                    safety priority (e.g. living alone -> higher w_bbs)
NOTE: this mapping is a clinician-reviewable starting point, not a
validated clinical algorithm; it is the explicit object of the proposed
clinician-feedback iteration described in the accompanying manuscript.

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - Persona is a frozen dataclass instead of a MATLAB struct built by a
    switch/case. MATLAB's "missing field" check (isfield over a
    hardcoded list of 17 required field names) has no Python
    equivalent needed: a dataclass literally cannot be constructed
    without all its fields, so that check is structurally guaranteed
    rather than runtime-validated. The two SUBSTANTIVE data-validity
    checks MATLAB performs -- ais_probs sums to 1, therapy_efficacy has
    exactly 13 entries -- are real numeric invariants Python's type
    system does NOT guarantee, so those are kept, but run ONCE for
    EVERY persona at module-IMPORT time (see _validate_persona calls
    below) rather than per-call like the MATLAB source. This is a
    genuine improvement, not just a literal-translation nicety: with
    18 personas hand-transcribed from MATLAB, eager validation catches
    a transcription typo in ANY of them immediately on import, not only
    when that specific persona happens to be requested.
  - define_persona() returns the SAME persona instance on every call
    for a given name (no defensive per-call copy), unlike MATLAB's
    struct value semantics which implicitly copy on every call. Nothing
    in the currently-ported pipeline mutates persona fields after
    construction, so this is a deliberate simplification, not an
    oversight -- see the project's "don't add defensive handling for
    scenarios that can't happen" convention.
  - MATLAB's `case 'default'` doesn't exist explicitly -- 'default' is
    handled by the `otherwise` branch alongside genuinely unrecognised
    names, distinguished only by whether a warning fires
    (`if ~strcmpi(name,'default')`). This port keeps that exact
    three-way distinction: a name matching one of the 17 explicit
    personas returns it silently; the literal string "default" returns
    the default persona silently; anything else warns AND falls back
    to the default persona.
"""

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from applications.sci_rehab.sindy_model_constants import PersonaWeights


@dataclass(frozen=True)
class Persona:
    ais_probs: np.ndarray          # (4,) P(AIS A/B/C/D), sums to 1
    age_range: Tuple[float, float]  # [min, max] raw age (years)
    dpi_range: Tuple[float, float]  # [min, max] days-post-injury at admission
    caregiver_alpha: float          # Beta-dist alpha for caregiver support
    caregiver_beta: float           # Beta-dist beta  for caregiver support
    motivation: float               # [0,1] amplifies response to therapy dose
    fatigue_tol: float              # [0,1] tolerance to high-load therapies
    pain_sensitivity: float         # [0,1] attenuates high-load response when high
    comorbidity: float              # [0,1] reduces functional ceiling (swat penalty)
    therapy_efficacy: np.ndarray    # (13,) per-therapy multiplier, ordered U1..U13
    w_scim: float
    w_bbs: float
    w_resource: float
    w_terminal_scim: float
    w_terminal_bbs: float
    name: str
    description: str


def persona_weights(persona: Persona) -> PersonaWeights:
    """Extract the 5 cost-function fields into a PersonaWeights instance
    for direct use with SindyModelConstants(persona=...) / l_cost.py."""
    return PersonaWeights(
        w_scim=persona.w_scim, w_bbs=persona.w_bbs,
        w_resource=persona.w_resource,
        w_terminal_scim=persona.w_terminal_scim,
        w_terminal_bbs=persona.w_terminal_bbs)


def _validate_persona(persona: Persona, source_name: str):
    if abs(float(np.sum(persona.ais_probs)) - 1.0) >= 1e-6:
        raise ValueError(
            f'define_persona: ais_probs must sum to 1 for persona "{source_name}"')
    if len(persona.therapy_efficacy) != 13:
        raise ValueError(
            'define_persona: therapy_efficacy must have 13 entries (U1..U13) '
            f'for "{source_name}"')


_PERSONAS = {

    # -- 1. High Motivator -------------------------------------------------
    #  Young to mid-age, strong internal drive, tolerates intensive therapy.
    #  Broad strong responder; peaks on active strengthening/gait/balance.
    "high_motivator": Persona(
        ais_probs=np.array([0.10, 0.15, 0.35, 0.40]),
        age_range=(20, 50), dpi_range=(10, 40),
        caregiver_alpha=3, caregiver_beta=2,
        motivation=0.90, fatigue_tol=0.80, pain_sensitivity=0.20, comorbidity=0.10,
        #                U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        therapy_efficacy=np.array([1.1, 1.0, 1.3, 1.2, 0.9, 1.0, 1.2, 1.3, 0.9, 1.0, 1.0, 1.2, 1.3]),
        w_scim=2.5, w_bbs=1.2, w_resource=5e-3,
        w_terminal_scim=60.0, w_terminal_bbs=30.0,
        name="High Motivator",
        description="Young/mid-age, high drive, tolerates intensive regimens"),

    # -- 2. Frail Elder ----------------------------------------------------
    #  Older adult, AIS A/B dominant, fatigue-limited, high comorbidity.
    #  Poor tolerance of high-load strengthening/NMES; thrives on balance,
    #  positioning, antispastic care and assistive-product training.
    "frail_elder": Persona(
        ais_probs=np.array([0.45, 0.30, 0.15, 0.10]),
        age_range=(60, 80), dpi_range=(20, 60),
        caregiver_alpha=2, caregiver_beta=3,
        motivation=0.45, fatigue_tol=0.25, pain_sensitivity=0.70, comorbidity=0.65,
        therapy_efficacy=np.array([1.0, 1.2, 0.6, 0.6, 1.2, 1.1, 1.3, 0.8, 1.3, 1.1, 1.2, 0.9, 0.7]),
        w_scim=1.5, w_bbs=2.0, w_resource=0.20,
        w_terminal_scim=35.0, w_terminal_bbs=45.0,
        name="Frail Elder",
        description="Older adult, AIS A/B, fatigue-limited, high comorbidity"),

    # -- 3. Young Athlete --------------------------------------------------
    #  Adolescent/young adult, high pre-injury fitness, rapid responder.
    #  Excels at high-load active therapy; less need for compensatory aids.
    "young_athlete": Persona(
        ais_probs=np.array([0.05, 0.10, 0.40, 0.45]),
        age_range=(16, 35), dpi_range=(5, 25),
        caregiver_alpha=2.5, caregiver_beta=2,
        motivation=0.95, fatigue_tol=0.95, pain_sensitivity=0.10, comorbidity=0.05,
        therapy_efficacy=np.array([1.1, 0.9, 1.5, 1.4, 0.8, 1.0, 1.2, 1.5, 0.7, 0.8, 0.9, 1.2, 1.5]),
        w_scim=3.0, w_bbs=1.5, w_resource=1e-3,
        w_terminal_scim=80.0, w_terminal_bbs=40.0,
        name="Young Athlete",
        description="Adolescent/young adult, high fitness pre-injury, rapid responder"),

    # -- 4. Low Social Support ---------------------------------------------
    #  Mixed AIS, minimal caregiver. Favours therapies that build durable
    #  independence in-clinic (assistive products, wheelchair/mobility,
    #  hand/arm) over ones needing home reinforcement (ROM, stretching).
    "low_support": Persona(
        ais_probs=np.array([0.25, 0.25, 0.25, 0.25]),
        age_range=(25, 65), dpi_range=(15, 55),
        caregiver_alpha=1.2, caregiver_beta=4.0,
        motivation=0.60, fatigue_tol=0.55, pain_sensitivity=0.40, comorbidity=0.30,
        therapy_efficacy=np.array([0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 1.0, 1.0, 1.3, 1.3, 1.0, 1.2, 1.0]),
        w_scim=2.0, w_bbs=1.0, w_resource=8e-2,
        w_terminal_scim=50.0, w_terminal_bbs=25.0,
        name="Low Social Support",
        description="Mixed AIS, minimal caregiver, limited home reinforcement"),

    # -- 5. Pain Sensitive -------------------------------------------------
    #  Chronic pain limits high-load escalation. Favours gentle therapies:
    #  positioning, antispastic care, stretching, functional positioning.
    "pain_sensitive": Persona(
        ais_probs=np.array([0.20, 0.35, 0.35, 0.10]),
        age_range=(30, 70), dpi_range=(20, 60),
        caregiver_alpha=2, caregiver_beta=2,
        motivation=0.55, fatigue_tol=0.40, pain_sensitivity=0.85, comorbidity=0.50,
        therapy_efficacy=np.array([1.0, 1.2, 0.6, 0.6, 1.3, 1.3, 1.0, 0.7, 1.1, 1.0, 1.2, 1.0, 0.7]),
        w_scim=1.8, w_bbs=1.0, w_resource=0.15,
        w_terminal_scim=40.0, w_terminal_bbs=20.0,
        name="Pain Sensitive",
        description="Chronic pain limits intensity escalation, AIS B/C"),

    # -- Maya (P1) -----------------------------------------------------
    #  24, C1-C4 AIS A, urban, strong parental caregiver support, university student.
    "p1_maya": Persona(
        ais_probs=np.array([0.94, 0.02, 0.02, 0.02]),
        age_range=(20, 28), dpi_range=(15, 45),
        caregiver_alpha=4, caregiver_beta=1.5,
        motivation=0.65, fatigue_tol=0.45, pain_sensitivity=0.45, comorbidity=0.40,
        therapy_efficacy=np.array([1.10, 1.30, 0.60, 1.20, 1.10, 1.10, 0.60, 0.50, 1.30, 1.20, 1.20, 1.30, 0.60]),
        w_scim=2.6, w_bbs=0.8, w_resource=0.04,
        w_terminal_scim=55, w_terminal_bbs=15,
        name="Maya (P1)",
        description="24, C1-C4 AIS A, urban, strong parental caregiver support, university student"),

    # -- Sophia --------------------------------------------------------
    #  36, C1-C4 AIS B, rural NB, assisted living, depression/anxiety, 110km from centre.
    "sophia": Persona(
        ais_probs=np.array([0.02, 0.94, 0.02, 0.02]),
        age_range=(32, 40), dpi_range=(20, 60),
        caregiver_alpha=1.3, caregiver_beta=4.0,
        motivation=0.40, fatigue_tol=0.40, pain_sensitivity=0.55, comorbidity=0.45,
        therapy_efficacy=np.array([1.00, 1.20, 0.50, 0.80, 1.20, 1.10, 1.00, 0.50, 1.30, 1.20, 1.20, 0.90, 0.50]),
        w_scim=2.0, w_bbs=1.0, w_resource=0.18,
        w_terminal_scim=35, w_terminal_bbs=20,
        name="Sophia",
        description="36, C1-C4 AIS B, rural NB, assisted living, depression/anxiety, 110km from centre"),

    # -- Riley (P3) ------------------------------------------------------
    #  17, C1-C4 AIS D, high pre-injury activity, lives with father, no comorbidities.
    "p3_riley": Persona(
        ais_probs=np.array([0.02, 0.02, 0.02, 0.94]),
        age_range=(15, 20), dpi_range=(5, 25),
        caregiver_alpha=3.5, caregiver_beta=1.5,
        motivation=0.85, fatigue_tol=0.85, pain_sensitivity=0.20, comorbidity=0.05,
        therapy_efficacy=np.array([1.00, 0.80, 1.30, 1.20, 0.70, 0.90, 1.30, 1.30, 0.70, 0.80, 0.80, 1.10, 1.20]),
        w_scim=2.8, w_bbs=1.6, w_resource=0.015,
        w_terminal_scim=65, w_terminal_bbs=35,
        name="Riley (P3)",
        description="17, C1-C4 AIS D, high pre-injury activity, lives with father, no comorbidities"),

    # -- Jean Guy --------------------------------------------------------
    #  58, C5-T1 AIS A, rural Acadian NB, diabetes/hypertension, 300km from centre, seasonal fisherman.
    "jean_guy": Persona(
        ais_probs=np.array([0.94, 0.02, 0.02, 0.02]),
        age_range=(54, 63), dpi_range=(20, 60),
        caregiver_alpha=2.2, caregiver_beta=2.5,
        motivation=0.55, fatigue_tol=0.40, pain_sensitivity=0.55, comorbidity=0.60,
        therapy_efficacy=np.array([1.00, 1.20, 0.50, 0.90, 1.20, 1.10, 0.70, 0.40, 1.30, 1.20, 1.20, 1.10, 0.50]),
        w_scim=2.2, w_bbs=0.8, w_resource=0.22,
        w_terminal_scim=40, w_terminal_bbs=15,
        name="Jean Guy",
        description="58, C5-T1 AIS A, rural Acadian NB, diabetes/hypertension, 300km from centre, seasonal fisherman"),

    # -- Daniel (P5) -------------------------------------------------
    #  42, C5-T1 AIS C, urban, high income, computer programmer, high pre- and post-injury activity.
    "p5_daniel": Persona(
        ais_probs=np.array([0.02, 0.02, 0.94, 0.02]),
        age_range=(38, 48), dpi_range=(10, 40),
        caregiver_alpha=3.5, caregiver_beta=1.5,
        motivation=0.90, fatigue_tol=0.80, pain_sensitivity=0.25, comorbidity=0.10,
        therapy_efficacy=np.array([1.00, 0.80, 1.30, 1.20, 0.80, 0.90, 1.20, 1.20, 0.80, 0.90, 0.90, 1.10, 1.40]),
        w_scim=2.8, w_bbs=1.4, w_resource=0.005,
        w_terminal_scim=70, w_terminal_bbs=35,
        name="Daniel (P5)",
        description="42, C5-T1 AIS C, urban, high income, computer programmer, high pre- and post-injury activity"),

    # -- Joan --------------------------------------------------------
    #  69, C5-T1 AIS D, suburban, recently widowed, living alone, retired academic.
    "joan": Persona(
        ais_probs=np.array([0.02, 0.02, 0.02, 0.94]),
        age_range=(65, 73), dpi_range=(10, 40),
        caregiver_alpha=1.5, caregiver_beta=4.0,
        motivation=0.60, fatigue_tol=0.55, pain_sensitivity=0.20, comorbidity=0.20,
        therapy_efficacy=np.array([1.00, 1.10, 0.80, 0.90, 1.00, 1.00, 1.30, 1.00, 1.20, 1.10, 1.10, 1.00, 0.80]),
        w_scim=2.0, w_bbs=2.0, w_resource=0.1,
        w_terminal_scim=50, w_terminal_bbs=50,
        name="Joan",
        description="69, C5-T1 AIS D, suburban, recently widowed, living alone, retired academic"),

    # -- Kenny --------------------------------------------------------
    #  27, T2-T12 AIS A, rural, FASD/ADHD, low income, limited provider access, strong community ties.
    "kenny": Persona(
        ais_probs=np.array([0.94, 0.02, 0.02, 0.02]),
        age_range=(23, 32), dpi_range=(15, 45),
        caregiver_alpha=3.0, caregiver_beta=2.0,
        motivation=0.65, fatigue_tol=0.65, pain_sensitivity=0.40, comorbidity=0.35,
        therapy_efficacy=np.array([1.00, 0.90, 1.20, 1.00, 0.80, 0.90, 1.00, 0.60, 1.30, 1.40, 1.00, 1.20, 0.90]),
        w_scim=2.4, w_bbs=0.8, w_resource=0.2,
        w_terminal_scim=50, w_terminal_bbs=15,
        name="Kenny",
        description="27, T2-T12 AIS A, rural, FASD/ADHD, low income, limited provider access, strong community ties"),

    # -- Marcus (P8) -------------------------------------------------
    #  25, T2-T12 AIS B, fused mid-back, construction foreman, no caregiver/family support, high substance use.
    "p8_marcus": Persona(
        ais_probs=np.array([0.02, 0.94, 0.02, 0.02]),
        age_range=(22, 30), dpi_range=(10, 35),
        caregiver_alpha=1.2, caregiver_beta=4.5,
        motivation=0.70, fatigue_tol=0.75, pain_sensitivity=0.35, comorbidity=0.30,
        therapy_efficacy=np.array([1.00, 1.00, 1.20, 1.20, 0.90, 1.00, 1.10, 1.00, 1.30, 1.30, 1.00, 1.10, 1.10]),
        w_scim=2.4, w_bbs=1.2, w_resource=0.12,
        w_terminal_scim=55, w_terminal_bbs=25,
        name="Marcus (P8)",
        description="25, T2-T12 AIS B, fused mid-back, construction foreman, no caregiver/family support, high substance use"),

    # -- Aaliyah (P9) -------------------------------------------------
    #  32, T2-T12 AIS D, Brown-Sequard syndrome, diabetes, lives with roommate, active online community.
    "p9_aaliyah": Persona(
        ais_probs=np.array([0.02, 0.02, 0.02, 0.94]),
        age_range=(28, 36), dpi_range=(10, 35),
        caregiver_alpha=1.8, caregiver_beta=3.0,
        motivation=0.80, fatigue_tol=0.55, pain_sensitivity=0.30, comorbidity=0.30,
        therapy_efficacy=np.array([1.00, 1.00, 1.00, 0.90, 0.90, 1.00, 1.30, 1.20, 0.90, 1.00, 1.00, 1.10, 1.00]),
        w_scim=2.4, w_bbs=1.6, w_resource=0.1,
        w_terminal_scim=55, w_terminal_bbs=35,
        name="Aaliyah (P9)",
        description="32, T2-T12 AIS D, Brown-Sequard syndrome, diabetes, lives with roommate, active online community"),

    # -- Pierre (P10) -------------------------------------------------
    #  40, L1-S5 AIS A, Quebec, electrician, heart condition, smoker, no primary care access.
    "p10_pierre": Persona(
        ais_probs=np.array([0.94, 0.02, 0.02, 0.02]),
        age_range=(36, 45), dpi_range=(20, 60),
        caregiver_alpha=3.0, caregiver_beta=2.0,
        motivation=0.50, fatigue_tol=0.40, pain_sensitivity=0.40, comorbidity=0.55,
        therapy_efficacy=np.array([1.00, 0.90, 1.10, 1.00, 0.70, 0.90, 0.90, 0.70, 1.20, 1.40, 1.00, 1.20, 0.60]),
        w_scim=2.4, w_bbs=0.8, w_resource=0.16,
        w_terminal_scim=50, w_terminal_bbs=15,
        name="Pierre (P10)",
        description="40, L1-S5 AIS A, Quebec, electrician, heart condition, smoker, no primary care access"),

    # -- Niran (P11) -------------------------------------------------
    #  75, L1-S5 AIS C, dementia, lives with adult children, in-home caregiver, low baseline activity.
    "p11_niran": Persona(
        ais_probs=np.array([0.02, 0.02, 0.94, 0.02]),
        age_range=(70, 80), dpi_range=(15, 50),
        caregiver_alpha=3.5, caregiver_beta=1.5,
        motivation=0.40, fatigue_tol=0.30, pain_sensitivity=0.40, comorbidity=0.65,
        therapy_efficacy=np.array([1.20, 1.20, 0.60, 0.70, 1.20, 1.10, 0.70, 0.50, 1.20, 1.00, 1.20, 0.80, 0.50]),
        w_scim=1.8, w_bbs=1.2, w_resource=0.14,
        w_terminal_scim=35, w_terminal_bbs=20,
        name="Niran (P11)",
        description="75, L1-S5 AIS C, dementia, lives with adult children, in-home caregiver, low baseline activity"),

    # -- Lucia (P12) -------------------------------------------------
    #  65, L1-S5 AIS D, posterior cord syndrome, hypertension/obesity, widow, lives alone, retired librarian.
    "p12_lucia": Persona(
        ais_probs=np.array([0.02, 0.02, 0.02, 0.94]),
        age_range=(60, 70), dpi_range=(15, 50),
        caregiver_alpha=1.6, caregiver_beta=3.5,
        motivation=0.55, fatigue_tol=0.40, pain_sensitivity=0.50, comorbidity=0.50,
        therapy_efficacy=np.array([1.00, 1.10, 0.70, 0.80, 1.10, 1.00, 1.40, 0.90, 1.10, 1.00, 1.10, 0.90, 0.60]),
        w_scim=1.8, w_bbs=1.8, w_resource=0.15,
        w_terminal_scim=40, w_terminal_bbs=40,
        name="Lucia (P12)",
        description="65, L1-S5 AIS D, posterior cord syndrome, hypertension/obesity, widow, lives alone, retired librarian"),
}

# -- Default (population average) -- MATLAB's `otherwise` fallback, not
# an explicit `case` -- see module docstring's three-way distinction.
_DEFAULT_PERSONA = Persona(
    ais_probs=np.array([0.30, 0.20, 0.25, 0.25]),
    age_range=(18, 80), dpi_range=(10, 60),
    caregiver_alpha=2, caregiver_beta=2,
    motivation=0.70, fatigue_tol=0.60, pain_sensitivity=0.30, comorbidity=0.25,
    therapy_efficacy=np.ones(13),
    w_scim=2.0, w_bbs=1.0, w_resource=1e-2,
    w_terminal_scim=50.0, w_terminal_bbs=25.0,
    name="Default",
    description="Original population-average parameters (no persona)")

for _key, _p in {**_PERSONAS, "default": _DEFAULT_PERSONA}.items():
    _validate_persona(_p, _key)


def define_persona(name: str) -> Persona:
    key = name.strip().lower()
    if key in _PERSONAS:
        return _PERSONAS[key]
    if key != "default":
        warnings.warn(f'define_persona: unknown persona "{name}". Using default.')
    return _DEFAULT_PERSONA
