"""
Print the SINDy-identified governing equations in clinician-readable
form, stripped of normalisation scaling.

Ported from Katie's CODES/KATIE'S CODE/.../print_identified_equations.m.
Last file in the SINDy pipeline (build_sindy_library.py through
compute_fisher_information.py).

Implements the clinician-interpretability requirement from the
manuscript's Section IV-E: the sparse differential equations are
printed with physically meaningful variable names and coefficient
magnitudes, so physiotherapists can inspect, critique, and override the
inferred patient dynamics -- not just trust a coefficient matrix.

For each of the two dynamic states (SCIM, BBS -- static states are
already zeroed by main_sindy_pipeline.m's "zero static state equations"
step and are skipped here, same min(2, n_states) convention as
prune_sindy_bic.py), prints the equation with signed coefficients, plus
a short "clinical interpretation" line translating the top 2 dominant
terms into plain language (a7 -> U7, x1 -> SCIM, ...) and whether each
one increases or decreases the state. Closes with an overall sparsity
summary across both dynamic-state equations.

INPUTS
  Xi          (n_terms, n_states)  sparse coefficients, PHYSICAL units
              (as produced by stlsq.py / prune_sindy_bic.py -- already
              de-normalised, used directly, no further scaling here)
  lib_labels  list[str] of length n_terms  library term labels with
              "[s=...]" scale tags (as produced by build_sindy_library.py)
  state_names list[str] of length n_states
  cfg         unused (see deviation note)

OUTPUT
  report  str  the full printed report (also printed to stdout as a
          side effect, matching the MATLAB source's fprintf calls) --
          returning it too is a minor testability addition beyond the
          literal MATLAB signature (which has no return value), not a
          behaviour change.

DEVIATION FROM THE LITERAL MATLAB SOURCE:
  - `cfg` is genuinely unused in the MATLAB source's own function body
    -- despite the docstring header listing it as "(.verbose)", nothing
    ever checks cfg.verbose; the function always prints. Kept as an
    unused parameter for call-site fidelity, same convention as
    Measurement.m's unused u/c.
  - COMMENT/CODE MISMATCH FOUND IN THE MATLAB SOURCE (flagged, not
    silently corrected): the comment above the a(\\d+) -> U$1
    substitution claims it uses a "word boundary so that a1 does not
    partially match a10..a13" -- but the actual MATLAB regex has NO
    word-boundary anchor there (unlike the x1..x6 substitutions, which
    DO use MATLAB's `\\>` end-of-word anchor). The substitution is
    still CORRECT in practice, just not for the stated reason: `\\d+`
    is greedy, so a(\\d+) applied to "a10" matches the whole "a10" in
    one shot (capturing "10"), never stopping early at "a1" -- no
    boundary anchor was ever needed for this specific pattern. Ported
    literally: no boundary on the a(\\d+) substitution, matching
    MATLAB's `\\>` with Python's `\\b` only on the x1..x6 substitutions.
  - Uses re.sub(r"\\s*\\[s=[\\d.]+\\]", "", label) to strip the scale tag,
    matching the idiom already used consistently in
    build_sindy_library.py/prune_sindy_bic.py, rather than the MATLAB
    source's own capture-and-extract `regexp(...,'tokens')` mechanism
    used only in THIS file -- same operation, equivalent result; the
    MATLAB source itself is inconsistent about which of the two
    mechanisms it uses across different files for this identical task.
  - Two DIFFERENT "is this coefficient zero" checks are preserved AS
    TWO DIFFERENT CHECKS, matching the MATLAB source exactly rather
    than being unified: `abs(xi_s) > 1e-8` (a numerical-noise floor)
    decides which terms to print per equation, while the final summary
    line uses exact `!= 0` (np.count_nonzero) for "how sparse is this
    model". These agree in practice (stlsq.py/prune_sindy_bic.py
    already produce exact zeros for pruned terms), but they're
    genuinely different criteria in the source and are kept that way.
  - Exact fprintf whitespace/blank-line layout is not reproduced
    byte-for-byte -- this is a human-readable diagnostic report, not
    something anything downstream parses, so the port preserves the
    same INFORMATION and overall structure (banner, one block per
    dynamic-state equation, sparsity summary) without chasing exact
    MATLAB whitespace.
"""

import re

import numpy as np

_BANNER = "══════════════════════════════════════════════════════════════════"


def print_identified_equations(Xi, lib_labels, state_names, cfg):
    n_terms, n_states = Xi.shape

    raw_labels = [_strip_scale_tag(lbl) for lbl in lib_labels]

    lines = ["", _BANNER, "  SINDy Identified Equations (physical coefficients)", _BANNER]

    dynamic_states = range(min(2, n_states))

    for s in dynamic_states:
        xi_s = Xi[:, s]
        nonzero = np.where(np.abs(xi_s) > 1e-8)[0]

        lines.append("")
        if nonzero.size == 0:
            lines.append(f"  d({state_names[s]})/dt = 0  (all coefficients zeroed)")
            continue

        terms_str = "  ".join(f"{xi_s[idx]:+.4f} * {raw_labels[idx]}" for idx in nonzero)
        lines.append(f"  d({state_names[s]})/dt = {terms_str}")

        interpretation = _clinical_interpretation(
            [raw_labels[idx] for idx in nonzero], xi_s[nonzero], state_names[s])
        lines.append(f"    [Active terms: {nonzero.size} / {n_terms}  |  {interpretation}]")

    lines.append("")
    Xi_dyn = Xi[:, :min(2, n_states)]
    nnz = np.count_nonzero(Xi_dyn)
    total = Xi_dyn.size
    sparsity = 100 * (1 - nnz / total) if total > 0 else 0.0
    lines.append(f"  Non-zero coefficients (dynamic states): {nnz} / {total}  "
                  f"(sparsity: {sparsity:.1f}%)")
    lines.append(_BANNER)
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


def _strip_scale_tag(label):
    return re.sub(r"\s*\[s=[\d.]+\]", "", label)


def _clinical_interpretation(labels, coeffs, state_name):
    sort_idx = np.argsort(-np.abs(coeffs))
    top_n = min(2, len(labels))

    parts = []
    for k in range(top_n):
        idx = sort_idx[k]
        lbl = labels[idx]
        coeff = coeffs[idx]
        direction = "increases" if coeff >= 0 else "decreases"

        # therapies a1..a13 -> WHO codes U1..U13 (see module docstring
        # re: the "word boundary" comment/code mismatch in the MATLAB
        # source -- \d+'s greediness already prevents a1/a10 confusion)
        lbl_clean = re.sub(r"\^2", " squared", lbl)
        lbl_clean = re.sub(r"a(\d+)", r"U\1", lbl_clean)
        lbl_clean = re.sub(r"x1\b", "SCIM", lbl_clean)
        lbl_clean = re.sub(r"x2\b", "BBS", lbl_clean)
        lbl_clean = re.sub(r"x3\b", "AIS", lbl_clean)
        lbl_clean = re.sub(r"x4\b", "Age", lbl_clean)
        lbl_clean = re.sub(r"x5\b", "DPI", lbl_clean)
        lbl_clean = re.sub(r"x6\b", "Caregiver", lbl_clean)

        parts.append(f"{lbl_clean} {direction} {state_name}")

    return "; ".join(parts)
