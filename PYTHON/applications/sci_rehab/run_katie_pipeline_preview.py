"""
Cheap dry run of run_katie_pipeline.py -- same PART A (identification) /
PART B (dual control) structure, drastically smaller dimensions (fewer
synthetic patients/sessions, smaller CV grid, fewer iLQG/dual-control
iterations). Mirrors the MATLAB source's own run_sgba_personas_preview.m
convention: confirm the full pipeline composes and runs end-to-end
without errors before committing to the full-scale run_katie_pipeline.py
(which takes materially longer -- 60 patients, full CV grid, longer
control horizon).

Not a correctness benchmark on its own (n_patients=12 is too few for a
tight SINDy fit / meaningful NMAE) -- just a fast smoke test that every
stage (data gen -> SINDy ID -> validation -> Fisher info -> dual control)
still runs and produces sane shapes/values.

Run directly: python run_katie_pipeline_preview.py
"""

import numpy as np

from extensions.sindy.print_identified_equations import print_identified_equations
from applications.sci_rehab.run_katie_pipeline import (
    print_control_report,
    run_dual_control_planning,
    run_identification,
    select_uncertain_terms,
)

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ident = run_identification("frail_elder", n_patients=12, n_sessions=6,
                                  noise_std=0.03, rng=rng, verbose=True)

    print_identified_equations(ident.Xi, ident.lib_labels, ident.dataset.state_names, cfg={})

    xi_idx_pairs = select_uncertain_terms(ident.Xi, ident.protected_idx, top_k=1)

    result, p_true, swat = run_dual_control_planning(
        ident, xi_idx_pairs, n_sessions=6, max_du_iterations=20,
        first_run_max_du_iterations=40, rng=rng, verbose=False)

    print_control_report(ident, result, xi_idx_pairs, p_true, swat)

    print("=== PREVIEW RUN COMPLETE -- pipeline composes end-to-end. ===")
