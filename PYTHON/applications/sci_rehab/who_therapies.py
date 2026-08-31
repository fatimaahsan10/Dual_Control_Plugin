"""
Central definition of the 13 WHO Package-of-Interventions therapy
control signals.

Ported from Katie's CODES/KATIE'S CODE/.../who_therapies.m.

Replaces the older 3 abstract controls (Intensity/Modality/Frequency)
with 13 concrete rehabilitation interventions drawn from the WHO
"Package of Interventions for Rehabilitation" (Spinal Cord Injury
module). Each control signal u_j in [0, 1] is the DOSE of that therapy
prescribed in a given session (0 = not used, 1 = maximum dose).

Single source of truth consumed by (once ported): generate_synthetic_dataset.py
(ground-truth dynamics). Already consumed indirectly by
define_persona.py (therapy_efficacy is ordered U1..U13 to match this
file's column order) and print_identified_equations.py (a1..a13 -> U1..U13
label translation).

FUNCTIONAL OUTCOME MODEL
Each therapy contributes to the two dynamic functional states:
    G[0, :]  SCIM (independence/motor function) gain per unit dose
    G[1, :]  BBS  (balance) gain per unit dose
These are POPULATION-AVERAGE "true" gains -- the primary clinical
target of each therapy (e.g. balance training loads BBS heavily,
hand/arm training loads SCIM and almost nothing on balance).
define_persona.py's therapy_efficacy then scales these per persona so
the optimiser recommends a genuinely different therapy mix per patient
type -- see that module's docstring.

`load`: physical demand of each therapy in [0, 1]. High-load therapies
are meant to be attenuated for fatigue-limited/pain-sensitive personas
and incur marginally higher resource cost, once
generate_synthetic_dataset.py consumes it (not used by any currently-
ported file).

DEVIATION FROM THE LITERAL MATLAB SOURCE: none functionally -- this is
pure static data. who_therapies() still returns a FRESH WhoTherapies
instance on every call (matching MATLAB's per-call struct-construction
value semantics), so callers can't accidentally corrupt shared state by
mutating what they got back.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class WhoTherapies:
    codes: List[str]       # ["U1", ..., "U13"]
    names: List[str]        # human-readable therapy names
    assessment: List[str]   # WHO assessment-domain grouping (reporting only)
    G: np.ndarray             # (2, 13): row 0 = SCIM gain/dose, row 1 = BBS gain/dose
    load: np.ndarray          # (13,) physical demand in [0, 1]
    n: int                     # 13


def who_therapies() -> WhoTherapies:
    codes = [f"U{i}" for i in range(1, 14)]

    names = [
        "Range of motion exercises",                                      # U1
        "Positioning for contracture prevention (incl. weight bearing)",  # U2
        "Muscle-strengthening exercises",                                 # U3
        "Neuromuscular / functional electrical stimulation (NMES/FES)",   # U4
        "Antispastic pattern positioning",                                # U5
        "Stretching",                                                     # U6
        "Balance training",                                               # U7
        "Gait training",                                                  # U8
        "Assistive products for mobility (provision + training)",        # U9
        "Mobility training (incl. wheelchair skills)",                    # U10
        "Functional positioning",                                         # U11
        "Functional training for hand and arm use",                      # U12
        "Fitness training",                                               # U13
    ]

    # WHO assessment-domain grouping, for grouped reporting only.
    assessment = [
        "Joint mobility", "Joint mobility", "Joint mobility", "Joint mobility",
        "Muscle tone", "Muscle tone",
        "Balance",
        "Gait & walking", "Gait & walking",
        "Mobility", "Mobility",
        "Hand & arm use",
        "Exercise capacity",
    ]

    # base functional gains G = [SCIM; BBS] (2 x 13)
    #                       U1    U2    U3    U4    U5    U6    U7    U8    U9    U10   U11   U12   U13
    G_scim = np.array([0.15, 0.10, 0.30, 0.28, 0.08, 0.12, 0.10, 0.30, 0.22, 0.25, 0.15, 0.26, 0.18])
    G_bbs = np.array([0.08, 0.05, 0.15, 0.08, 0.06, 0.08, 0.35, 0.25, 0.06, 0.08, 0.10, 0.02, 0.15])
    G = np.vstack([G_scim, G_bbs])

    # physical load (demand) per therapy in [0, 1] -- strengthening,
    # NMES, gait, balance and fitness are the most demanding
    #                     U1    U2    U3    U4    U5    U6    U7    U8    U9    U10   U11   U12   U13
    load = np.array([0.30, 0.15, 0.85, 0.70, 0.10, 0.25, 0.55, 0.80, 0.20, 0.40, 0.15, 0.45, 0.75])

    return WhoTherapies(codes=codes, names=names, assessment=assessment,
                          G=G, load=load, n=len(codes))
