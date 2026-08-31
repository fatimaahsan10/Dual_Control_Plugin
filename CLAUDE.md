# Project: Curiosity-Driven AI for Spinal Cord Injury
Mitacs internship, University of New Brunswick.

## Goal
Apply curiosity-driven AI (intrinsic motivation / curiosity-driven RL) 
to spinal cord injury [rehab / motor recovery modeling — narrow this 
down further as you read more].

## Codebases
- todorov/ (MATLAB source: CODES/TODOROV'S CODE/; Python port: 
  PYTHON/core/ddp_solver/ — see "Repository layout" below): Tassa, 
  Mansard & Todorov (2014) control-limited DDP/iLQG toolbox — 
  deterministic, general-purpose optimal control library. NOT COVID/SCI-
  specific — pure generic algorithm (demos: car parking, simple linear 
  system).
- andrew/ (MATLAB source: CODES/ANDREW'S CODE/; Python port: 
  PYTHON/extensions/dual_control/ — see "Repository layout" below): 
  Dual/adaptive iLQG extension of Todorov's toolbox, applied 
  to SIDARTHE COVID-19 epidemic model as a testbed — confirmed 
  INTENTIONAL by supervisor. Adds stochastic dynamics, dual control 
  (uncertainty-aware), parameter estimation (SPKF), MPC receding horizon.
- CODES/KATIE'S CODE/ (folder: Curiosity_Algorithm_w_SINDy_V7_SGBA_personas): 
  received 2026-07-28, extracted from zip. MATLAB, ~46 files. Builds on 
  Andrew's iLQG/SPKF core (same files, attributed to Andrew Mathis in 
  headers, reused near-verbatim) but replaces the SIDARTHE COVID plant 
  with a genuine SCI-rehab state space (SCIM, BBS, AIS, Age, DPI, 
  Caregiver support), adds SINDy (sparse regression) to identify plant 
  dynamics from data instead of assuming them known, and adds a 
  13-therapy WHO "Package of Interventions for Rehabilitation" action 
  space plus 12 named SGBA+ (Sex- and Gender-Based Analysis Plus) 
  patient personas. Two entry-point chains: main_sindy_pipeline.m 
  (SINDy identification) -> Outer_Control_Loop.m (older 3-control 
  single-persona demo) / run_sgba_personas.m (current 12-persona, 
  13-therapy dual-iLQG pipeline, with run_sgba_personas_preview.m as a 
  cheap dry run). run_paper_results_v4.m is the current manuscript 
  results generator (supersedes run_paper_results.m). Not yet ported to 
  Python. Appears developed under GNU Octave (custom shims for 
  sgtitle/xline/yyaxis/crossvalind/randsample/betarnd), not licensed 
  MATLAB + Statistics Toolbox.
- zahid/: no code received/found (paper only — see "STATUS — Zahid 
  conversion" below; the Python code is an original implementation 
  derived from the paper's equations, not a port — split across 
  PYTHON/extensions/constraints/ and PYTHON/applications/
  pendulum_constrained/, see "Repository layout" below)

## Repository layout (PYTHON/)
Reorganized 2026-08-17 from four flat, per-person directories 
(PYTHON/todorov/, andrew/, katie/, zahid/ — now deleted) into a 
core/extensions/applications layering shared across all four 
codebases, at the supervisor's request. Every directory below is a 
real Python package (__init__.py); cross-module imports are absolute 
package imports (e.g. `from core.ddp_solver.ilqg import ilqg`), not the 
old sys.path.append hack. PYTHON/pyproject.toml + `pip install -e .` 
(run once from PYTHON/) makes core/extensions/applications importable 
from anywhere, so `python run_katie_demo.py`-style direct-run 
instructions elsewhere in this file still work unchanged.

- PYTHON/core/ddp_solver/ — Todorov's deterministic DDP/iLQG solver 
  (box_qp.py, back_pass.py, forward_pass.py, ilqg.py). The shared 
  algorithmic core every other codebase builds on.
- PYTHON/extensions/dual_control/ — Andrew's stochastic/dual-control 
  extension (backward_pass.py, forward_pass.py, ilqg_function.py, 
  todorov_estimator.py, spkf_function.py, simulate_system.py, 
  make_pd.py, finite_difference.py, main_outer_control_loop.py). Not a 
  thin wrapper around core/ddp_solver/ in the import graph — it's an 
  independent, self-contained reimplementation extended for stochastic 
  dynamics + parameter estimation (see Katie conversion plan's diff 
  analysis below). Reused as-is by Katie's SCI problem at 6-state/
  13-action dims. main_outer_control_loop.py lives here (not a 
  per-person folder) because it takes dynamics/measurement/cost/
  continuous_dynamics as plant-agnostic callables — confirmed generic 
  by PYTHON/applications/sci_rehab/run_katie_pipeline.py reusing it 
  unmodified.
- PYTHON/extensions/sindy/ — Katie's SINDy system-identification 
  pipeline (build_sindy_library.py, stlsq.py, cross_validate_lambda.py, 
  prune_sindy_bic.py, validate_sindy_model.py, 
  compute_fisher_information.py, print_identified_equations.py). 
  Plant-agnostic except validate_sindy_model.py, which imports 
  build_sindy_row from applications/sci_rehab/continuous_state_dynamics.py 
  — a real inversion of the core->extensions->applications layering, 
  inherited from the original code (reuses that helper rather than a 
  third copy) and flagged in validate_sindy_model.py's own imports 
  rather than silently normalized.
- PYTHON/extensions/constraints/ — Zahid's state-dependent constraint-
  bound extension to the core DDP solver (dynamic_control_bounds.py, 
  relative_degree_reduction.py).
- PYTHON/applications/sci_rehab/ — Katie's SCI plant/cost/personas and 
  demos (continuous_state_dynamics.py, discrete_state_dynamics.py, 
  measurement.py, l_cost.py, sindy_model_constants.py, who_therapies.py, 
  define_persona.py, generate_synthetic_dataset.py, run_katie_demo.py, 
  run_katie_pipeline.py, run_katie_pipeline_preview.py), plus the five 
  test_*_dims.py files that re-verify extensions/dual_control's reused 
  solver core at Katie's own 6-state/13-action dimensions 
  (test_core_solver_dims.py, test_estimator_ilqg_dims.py, 
  test_spkf_function_dims.py, test_simulate_system_dims.py, 
  test_utility_dims.py).
- PYTHON/applications/pendulum_constrained/ — Zahid's inverted-pendulum 
  demo reproducing the paper's own worked example (run_zahid_demo.py).
- PYTHON/applications/todorov_toy/ — Todorov's own generic demo 
  (run_todorov_demo.py; car parking / linear system).
- PYTHON/applications/linear_dual_control/ — Andrew's/the thesis's Sec. 
  3.4 linear dual-control toy example (linear_plant.py + 
  run_linear_demo.py; see "STATUS — linear dual-control example" below). 
  Added 2026-08-26 — Andrew's first applications/ entry; the SIDARTHE 
  COVID plant itself is still not ported (see "NOT YET PORTED" below), 
  so extensions/dual_control/ remains otherwise unaccompanied by any 
  SIDARTHE-specific application code.
- PYTHON/applications/dynamic_pricing/ — single-seller dynamic-pricing 
  plug-in (pricing_constants.py/continuous_dynamics.py/dynamics.py/
  measurement.py/cost.py/run_pricing_demo.py/test_plugin_contract.py), 
  reusing extensions/dual_control/main_outer_control_loop.py to learn an 
  unknown linear-demand price sensitivity `b` online. BACKFILLED into 
  this file 2026-08-27 — the code's own docstrings reference "CLAUDE.md's 
  dynamic-pricing math review"/"demo review" sections that were never 
  actually written here (a real doc gap from whatever session built it, 
  left unreconciled beyond this note — see cost.py's own REVISION 
  comment and run_pricing_demo.py's REVISION HISTORY comment for the 
  substance of those missing reviews: (a) pricing profit off the LAGGED 
  demand state gives price zero within-period cost under receding-horizon 
  MPC, so it pins at the control bound regardless of belief -- fixed by 
  pricing off the CURRENT price-implied demand D(u;b_hat) instead; (b) 
  u_lim_method=2's tanh-squash can saturate the raw control under a badly 
  wrong prior, trapping price even after the belief corrects -- fixed by 
  switching to u_lim_method=1 (box-QP bounds)). Both lessons are exactly 
  what applications/duopoly_pricing/ applies from the start rather than 
  rediscovering.
- PYTHON/applications/duopoly_pricing/ — TWO-seller generalization of 
  dynamic_pricing/ (duopoly_constants.py/continuous_dynamics.py/
  dynamics.py/measurement.py/cost.py/run_duopoly_demo.py/
  test_plugin_contract.py; see "STATUS — duopoly pricing" below). Added 
  2026-08-27, per explicit instruction as a new, separate folder with 
  ZERO changes to core/ddp_solver/protocols.py or 
  extensions/dual_control/protocols.py (confirmed: neither file was 
  touched, and the full PYTHON/ test suite passes 134/134 with this 
  folder's one new test included, zero regressions).
- PYTHON/applications/robot_arm_2link/ — 2-link planar robot arm (arm_
  constants.py/continuous_dynamics.py/dynamics.py/measurement.py/
  cost.py/test_plugin_contract.py/run_robot_arm_demo.py; see "STATUS — 
  2-link robot arm" below). Added 2026-08-30, proposed as math/mapping 
  first (no code) and built only after review, same "review before 
  build" workflow as every STATUS entry below that names a proposal 
  discussion. Standard rigid-body manipulator-equation plant (Spong/
  Hutchinson/Vidyasagar; Slotine & Li's classic "unknown payload mass" 
  adaptive-control benchmark) — genuinely different from every other 
  application/ plug-in so far: a real coupled-nonlinear-ODE robotics 
  problem (inertia matrix inversion, Coriolis/centrifugal terms, 
  gravity), not a linear/near-linear economic or biomedical plant. Zero 
  changes to core/ddp_solver/protocols.py or 
  extensions/dual_control/protocols.py (confirmed: neither file touched, 
  full PYTHON/ suite passes 262/262 with this folder's one new test 
  included, zero regressions).
- CODES/ (MATLAB source trees) and NOTES/ (diagrams) are untouched by 
  this reorganization — reference material, not part of the PYTHON/ 
  package tree.
- PYTHON/wizard/ — Streamlit guided model-building wizard (added 
  2026-08-27; see "STATUS — model-building wizard" below). App 
  orchestration only in app.py; all logic lives in independently-tested 
  modules (schema.py, equation_parser.py, generic_plant.py, storage.py, 
  validation_runner.py, solver_runner.py, plotting.py, 
  codegen_preview.py), plus models/ (user-saved configs, JSON) and 
  examples/ (two shipped reference configs reproducing dynamic_pricing/ 
  and duopoly_pricing/ through the wizard's own generic schema). A NEW, 
  generic layer that CALLS core/ddp_solver and extensions/dual_control 
  unmodified — zero changes to either package's protocols.py or any 
  other file in them.

## STATUS — Todorov conversion: COMPLETE ✅
All core files ported to Python and tested. Now live in 
PYTHON/core/ddp_solver/ (see "Repository layout" above):
- box_qp.py, back_pass.py, forward_pass.py, ilqg.py
Verified against an independently-derived LQR solution (exact match).

## STATUS — Andrew conversion: COMPLETE ✅
All 7 core files ported to Python and tested (24/24 tests passing). Now 
live in PYTHON/extensions/dual_control/ (see "Repository layout" above):
- forward_pass.py, backward_pass.py, Todorov_estimator.py, 
  iLQG_function.py, makePD.py, SPKF_function.py, 
  Main_Outer_Control_Loop.py
Verified end-to-end: deliberately wrong parameter (0.4) correctly 
converges toward true value (1.0) over iterations while cost drops — 
confirms dual-control behavior genuinely works, not just a trivial pass.

Important: Todorov's and Andrew's Python code are separate, independent 
codebases (not merged into one combined system) — matches supervisor's 
chosen strategy (Option A: convert individually, not combine-then-convert).

## STATUS — Katie conversion: IN PROGRESS (started 2026-07-29)
Python code was originally written in PYTHON/katie/; as of the 
2026-08-17 reorganization it's split by role between 
PYTHON/extensions/sindy/ (the plant-agnostic SINDy pipeline files) and 
PYTHON/applications/sci_rehab/ (the SCI plant/cost/persona/demo files 
plus the dims-verification tests below) — see "Repository layout" 
above.

**Reuse-as-is files confirmed at Katie's actual dimensions (6 states, 
13 actions)**, not just diffed on paper — each has a dedicated 
PYTHON/applications/sci_rehab/test_*_dims.py that imports 
extensions/dual_control's module directly (no duplication) and re-runs 
its correctness checks at n=6/m=13 instead of Andrew's small generic 
dims:
- backward_pass.py, forward_pass.py (test_core_solver_dims.py)
- todorov_estimator.py, ilqg_function.py (test_estimator_ilqg_dims.py)
- spkf_function.py (test_spkf_function_dims.py) — Katie's SPKF_function.m 
  is 466 lines vs Andrew's 165, but `Mode = 0` is hardcoded identically in 
  both (confirmed by reading both headers) and the extra lines are 
  entirely inside a dead `elseif Mode == 1` branch
- simulate_system.py (test_simulate_system_dims.py) — had no standalone 
  test at all before this, even at Andrew's own dimensions (only 
  exercised transitively through ilqg_function.py)
- box_qp.py, make_pd.py, finite_difference.py (test_utility_dims.py) — 
  quick checks, each driven at the dimension it's actually used at in 
  Katie's problem (box_qp at n=13 action dim, make_pd at n=6 state dim, 
  finite_difference at n=19 augmented state+action dim)

**Genuinely-new SCI plant/cost math ported and tested**, replacing 
Andrew's SIDARTHE plant:
- continuous_state_dynamics.py (from ContinuousStateDynamics.m) + 
  sindy_model_constants.py (SindyModelConstants/PersonaWeights 
  dataclasses — explicit substitute for MATLAB's `global SINDY_MODEL`, 
  passed as the existing `constants` argument already threaded through 
  forward_pass.py/simulate_system.py/todorov_estimator.py/ilqg_function.py) 
  — test_continuous_state_dynamics.py
- measurement.py (from Measurement.m) — test_measurement.py
- l_cost.py (from l_cost.m) — test_l_cost.py
- DISCREPANCY FOUND (flagged, not silently fixed): 
  ContinuousStateDynamics.m's header comment still says 
  "ACTION VECTOR (nu = 3): Intensity/Modality/Frequency" — stale, left 
  over from the older single-persona demo. The code itself is 
  action-dimension-generic; l_cost.m's own comment and 
  define_persona.m's `assert(numel(therapy_efficacy) == 13)` confirm the 
  real pipeline uses 13 WHO therapy doses. Noted in 
  continuous_state_dynamics.py's docstring rather than corrected in the 
  MATLAB source.
- discrete_state_dynamics.py (from DiscreteStateDynamics.m) — the 
  Euler-integration wrapper: x_{k+1} = x_k + dt*f(x_k,u_k) + F_a@w*sqrt(dt), 
  f = continuous_state_dynamics.py, F_a = blkdiag(F, Fp). This is what 
  actually satisfies the `dynamics()` callable contract 
  forward_pass.py/simulate_system.py expect (continuous_state_dynamics.py 
  alone doesn't). WIRING NOTE: F_a is (nx+n_p, nx+n_p), so nw must be 
  passed as nx+n_p when this is wired into forward_pass() later, not an 
  independently-chosen channel count like Andrew's SIDARTHE setup used. 
  RK4 branch (time_step_method=2) and the `lambda` parameter are both 
  dead in the MATLAB source (hardcoded to Euler; lambda's only use is a 
  commented-out line) — not ported, per the project's existing 
  confirmed-dead-branch convention (SPKF Mode==1, iLQG full_DDP). 
  test_discrete_state_dynamics.py (6 tests) plus an end-to-end sanity 
  check wiring it through the already-tested simulate_system.py/
  measurement.py (SCIM/BBS trajectories match hand-computed Euler steps 
  exactly, static states untouched). 
  ALL FOUR plant/cost files now ported: continuous_state_dynamics.py, 
  discrete_state_dynamics.py, measurement.py, l_cost.py.

**End-to-end demo (run_katie_demo.py, 2026-07-29):** wires the reused 
solver core (ilqg_function.py/simulate_system.py, imported directly, no 
duplication) to the four ported plant/cost files and runs dual-iLQG on 
one placeholder patient over 12 weekly sessions -- confirms the full 
stack actually composes, not just each piece in isolation. Two 
placeholders, both documented in the script's own docstring since 
neither exists yet: (1) Xi is built from who_therapies.m's population-
average G_scim/G_bbs gains (poly_order=1, linear-only) instead of a 
SINDy-identified model, since build_sindy_library.m/stlsq.m aren't 
ported; (2) persona=None -> l_cost.py's population-average default 
weights instead of a real define_persona.m persona. Also uses 
augment_states=True with n_p=0 (zero estimated parameters) to sidestep 
ilqg_function.py's augment_states=False branch, which assumes 
`constants` is a plain ndarray -- documented as a real constraint of 
that branch, not a bug workaround. 
NOTE: who_therapies.m's 13 gains sum to 2.49 (SCIM)/1.51 (BBS) if all 
13 therapies were dosed at max simultaneously -- enough to jump straight 
to the recovery ceiling in one session with nothing to stop it, since 
this linear placeholder has no diminishing-returns structure across a 
therapy mix (real SINDy identification + BIC pruning would be expected 
to produce a sparse handful of dominant terms instead). Scaled by a 
documented DEMO_GAIN_SCALE=0.08 so the demo shows week-over-week 
progress instead of saturating in one Euler step -- purely a 
placeholder-plant concession, not a claim about real per-therapy gains. 
Resulting trajectory (12 sessions, ceiling=0.85): SCIM ramps 0.20→0.86 
by session ~4-5 then holds at ceiling; BBS ramps more slowly, 0.15→0.85 
by session 12 (smaller who_therapies BBS gains); controller front-loads 
dose early (~full dose sessions 0-1) then tapers as it approaches the 
ceiling/cost tradeoff; top therapies by mean dose are U7 (balance 
training) and U8 (gait training) -- consistent with who_therapies.m's 
BBS-loading table. matplotlib installed 2026-07-29; chart saved to 
notes/katie_demo_chart.png.

## STATUS — Katie SINDy pipeline: COMPLETE ✅ (2026-07-29 to 2026-07-30)
Per the hand-port decision above (not pysindy). All 7 files ported and 
tested (65/65 tests passing across the full katie suite at the time). 
Now live in PYTHON/extensions/sindy/ (see "Repository layout" above): 
build_sindy_library.py, stlsq.py, cross_validate_lambda.py, 
prune_sindy_bic.py, validate_sindy_model.py, compute_fisher_information.py, 
print_identified_equations.py.

## STATUS — Katie personas: DONE (2026-07-30)
Files below now live in PYTHON/applications/sci_rehab/ (see "Repository 
layout" above).
who_therapies.py (from who_therapies.m) — static data only, 13 WHO 
Package-of-Interventions therapies: codes/names/assessment groupings, 
population-average base functional gains G (2x13: SCIM row, BBS row), 
and physical load per therapy. who_therapies() returns a FRESH instance 
per call (matches MATLAB's per-call struct-copy semantics -- mutating 
one caller's copy can't corrupt another's). test_who_therapies.py 
(3 tests): shapes/counts, spot-checked values against the MATLAB source 
(U7 balance training, U3 strengthening, U12 hand/arm), and 
no-shared-mutable-state between calls.

define_persona.py (from define_persona.m) — 18 personas total: 5 
population archetypes (high_motivator, frail_elder, young_athlete, 
low_support, pain_sensitive) + 12 named SGBA+ intersectional personas 
(p1_maya through p12_lucia, each representing ONE concrete patient, 
hence ais_probs ~94% concentrated on one AIS grade rather than spread) 
+ 1 default (population average, MATLAB's `otherwise` fallback, not an 
explicit `case`). therapy_efficacy (13 entries, ordered U1..U13 to 
match who_therapies.py) multiplies who_therapies.py's population-average 
gains per persona -- this is baked into the SIMULATED PHYSIOLOGY 
(consumed by generate_synthetic_dataset.py, not yet ported), not the 
cost function, so the controller genuinely discovers each persona's 
best therapy mix rather than following hand-coded per-persona rules. 
All 18 personas hand-transcribed from the MATLAB source into a Persona 
dataclass; MATLAB's runtime "missing field" check has no Python 
equivalent needed (a dataclass can't be constructed without all its 
fields), but the two SUBSTANTIVE numeric-invariant checks (ais_probs 
sums to 1, therapy_efficacy has 13 entries) are kept and run ONCE PER 
PERSONA AT MODULE-IMPORT TIME rather than per-call -- catches a 
transcription typo in ANY of the 18 immediately on import rather than 
only when that specific persona is requested (a real concern given how 
much manual numeric transcription this file involved; import succeeded 
clean, no typos found). 
Added persona_weights(persona) -> PersonaWeights bridging this file's 5 
cost-weight fields directly into the ALREADY-PORTED l_cost.py/
SindyModelConstants pipeline -- small, well-justified glue, not scope 
creep. 
define_persona() preserves MATLAB's exact three-way name-matching 
distinction: a name matching one of the 17 explicit personas returns it 
silently; the literal string "default" returns the default persona 
silently (MATLAB's `otherwise` branch specifically skips the warning 
for this one case via `if ~strcmpi(name,'default')`); anything else 
warns AND falls back to default. 
test_define_persona.py (7 tests): known-persona field values, 
case-insensitive/trimmed name matching, all 18 personas valid via the 
public API (not just the eager import-time check), unknown-name 
warning+fallback, the explicit-"default"-no-warning distinction (the 
trickiest part of the MATLAB logic to get right), persona_weights() 
end-to-end through SindyModelConstants -> l_cost.py, and confirmation 
that the 12 SGBA+ personas are AIS-concentrated (>=0.9) while the 5 
archetypes are spread (<0.9) -- directly testing the "one patient vs. 
population" distinction the docstring describes.

**run_katie_demo.py updated (2026-07-30) to use real personas.** Xi's 
per-therapy gains are now who_therapies().G * persona.therapy_efficacy 
(not just plain population-average gains), and constants.persona is a 
real persona_weights(define_persona(name)) instead of the None 
placeholder. __main__ now runs TWO personas back to back for direct 
comparison: "default" (reproduces the original placeholder-only demo 
exactly, since define_persona('default')'s weights match l_cost.py's 
own fallback and its therapy_efficacy is all 1.0) and "frail_elder". 
RESULT: genuinely different behavior, not just different numbers --  
default front-loads dose (0.999 mean dose week 1) and picks 
U7/U8/U13 (fitness) as top therapies, reaching the SCIM ceiling by 
session 5; frail_elder (w_resource=0.2, 20x more burden-averse; 
w_bbs=2.0>w_scim=1.5, unlike default's w_scim>w_bbs) doses far more 
conservatively (0.592 week 1, smooth taper) and swaps U13->U11 
(functional positioning) as its third pick -- traceable directly to 
frail_elder's therapy_efficacy: U13 (fitness) = 0.7 (poor responder) 
vs U11 = 1.2 (good responder). Confirms therapy_efficacy is doing real 
work on the recommended therapy MIX, not just the cost tradeoff.

## STATUS — generate_synthetic_dataset.py: DONE (2026-07-30)
The data generator behind the whole SINDy pipeline -- its X/A output 
(n_states,n_sessions,n_patients)/(n_actions,n_sessions,n_patients) is 
exactly what build_sindy_library.py expects, and it's what a real 
run_sgba_personas.m-style pipeline would call before 
build_sindy_library -> ... -> validate_sindy_model for each persona. 
True dynamics: dx1/dx2 = homeostatic decay+AIS interaction (STRONGLY 
saturated near the persona's recovery ceiling via head_i = 
clip(1-x_i/ceiling,0,1)) + persona-modulated therapy contribution 
(only MILDLY saturated via soft_i=0.55+0.45*head_i, so therapy effects 
stay SINDy-identifiable even near the ceiling). Per-therapy effective 
dose u_eff = u * therapy_efficacy * motivation * load_atten, where 
load_atten combines pain_sensitivity (penalizes high DOSES of 
high-LOAD therapies) and fatigue_tol (penalizes high-load therapies 
regardless of dose). Only 3-5 of 13 therapies are dosed per session 
(not all 13) -- decorrelates the dose regressors so SINDy can actually 
identify each therapy's effect individually from one-step finite 
differences; A(:,:,p) records the PRESCRIBED dose, never u_eff, so 
SINDy has to discover the persona-specific response purely from data, 
same as a real identification problem. 
COMMENT/CODE MISMATCH FOUND (flagged, not silently reconciled): the 
MATLAB file's top-level docstring describes the saturation as 
`sat_i = max(0, ceiling_adjusted - x_i)` (additive) but the actual 
simulation code implements `head_i = clip(1-x_i/ceiling,0,1)` (a 
ratio) -- materially different formulas. Ported the real code (the 
ratio form), which is what generate_synthetic_dataset.m actually runs. 
randsample/betarnd (MATLAB Stats Toolbox) replaced with 
numpy.random.Generator.choice/.beta per the existing decision in this 
file's Katie conversion plan; added an explicit `rng` parameter (not 
in the MATLAB signature) since MATLAB's implicit global RNG state has 
no equivalent to preserve -- same reasoning as 
cross_validate_lambda.py's `random_state`. 
Xi_true (diagnostic, approximate ground truth for coefficient-recovery 
checks) is built by calling build_sindy_library.py on a 2-patient 
slice purely to get matching label ordering, then hand-filling known 
coefficients at a representative mid-working-point -- doesn't capture 
the saturation/cross-term nonlinearity, so it's a rough reference, not 
exact ground truth (documented as such in the MATLAB source too). 
test_generate_synthetic_dataset.py (8 tests): shapes; exact 
reproducibility given the same rng seed; sparse per-session activation 
stays in [3,5]; terminal session repeats the previous dose; static 
states are byte-exact constant across sessions when noise_std=0; 
dynamic states and swat_ceiling stay in bounds; a hand-verified 
one-step dynamics check using a test-only "neutral" persona 
(motivation=1, pain_sensitivity=0, fatigue_tol=1 -> u_eff==u exactly) 
that reconstructs dx1/dx2 independently from the recorded X/A and 
compares bit-for-bit; and Xi_true's structure, including that zeroing 
one persona's therapy_efficacy entry zeroes exactly that action's two 
Xi_true coefficients (28 nonzero instead of the usual 30 = 
4 base terms + 2*13 action terms).

- build_sindy_library.py (from build_sindy_library.m) — builds the 
  SINDy regression problem (Theta candidate library + dX derivative 
  targets from patient trajectories), NOT the sparse solve itself 
  (that's stlsq.py, not yet ported). test_build_sindy_library.py 
  (5 tests): finite-difference correctness against a known-slope 
  trajectory, exact Theta-row hand computation for a small case 
  (poly_order=3, cross+trig on), protected_idx default (action terms) 
  + user terms, and shape/term-count checks at Katie's real dims 
  (6 states/13 actions -> 212 terms with poly_order=3+cross, matches 
  1+19+19+C(19,2)+2 by construction). 
  CORRECTNESS TRAP CAUGHT: MATLAB's `std(Theta,0,1)` defaults to N-1 
  (unbiased/sample) normalisation; NumPy's `.std()` defaults to N 
  (population, ddof=0). Used `ddof=1` explicitly -- getting this wrong 
  would have silently scaled every column's "[s=...]" tag (and, once 
  stlsq.py is ported, every threshold decision) by sqrt(N/(N-1)). 
  DISCREPANCIES FLAGGED (not silently fixed): (1) cfg['include_trig'] 
  is accessed without a default in build_sindy_library.py, matching the 
  MATLAB source's own unconditional access -- inconsistent with 
  ContinuousStateDynamics.m's OWN copy of a similar row-builder, which 
  DOES defensively check `isfield`. (2) The MATLAB docstring claims 
  specific cross-terms (x1*x3, x2*x3, a1*x3, x6*a1) are "prioritised" -- 
  the code just enumerates every pairwise cross-product uniformly; the 
  terms actually protected in main_sindy_pipeline.m ('x3*a3', 'x6*a8') 
  don't even match the docstring's list. Cosmetic doc drift in the 
  MATLAB source, not a functional bug.

- stlsq.py (from stlsq.m) — Sequential Thresholded Least Squares: the 
  sparse-regression solve itself (dX ~= Theta @ Xi), alternating ridge 
  least-squares refits with hard-thresholding of small normalised 
  coefficients until the support stabilises. Implements exactly the 
  three customisations that motivated hand-porting over pysindy (see 
  SINDy library decision above): `protected_idx` terms immune to 
  thresholding, `min_terms` floor (revives highest-magnitude dropped 
  unprotected terms), `max_terms` hard cap (greedily prunes smallest 
  unprotected active term, refitting after each removal). Same 
  ddof=1/N-1-std correctness trap as build_sindy_library.py, flagged 
  again in the module docstring since stlsq.m recomputes its own column 
  normalisation independently per call rather than reusing 
  build_sindy_library.m's stored tags (intentional design, already 
  documented there: keeps lambda thresholding on a consistent scale 
  regardless of dataset/fold size). 
  test_stlsq.py (5 tests, all noiseless/deterministic so exact support 
  recovery is checkable): exact sparse ground-truth recovery, protected 
  term survives an aggressive threshold that zeros an equally-tiny 
  unprotected term, max_terms cap prunes smallest-magnitude unprotected 
  first while protection overrides magnitude, min_terms floor revives 
  the highest-magnitude dropped terms, and an end-to-end composition 
  check running stlsq.py directly on build_sindy_library.py's output at 
  Katie's real dims (6 states/13 actions, 22-term cap, 3-term floor) -- 
  confirms default-protected action terms (a1..a13) never get zeroed.

- cross_validate_lambda.py (from cross_validate_lambda.m) — picks 
  stlsq's sparsity threshold lambda via 5-fold CV instead of a fixed 
  value: log-spaced grid (0.001-0.35, capped below 0.5 -- letting it 
  reach 0.5 previously let selection drift to the ceiling and strip 
  every action term), data-adaptive floor trimmed from a quick full-data 
  OLS fit, per-fold NMAE scoring with a fixed penalty for degenerate 
  folds (every unprotected term zeroed -- checked independently of 
  whether protected terms survived), a small complexity penalty (3% of 
  mean valid error) to break near-ties toward sparser models, and a 
  DELIBERATELY INVERTED 1-SE rule (picks the SMALLEST lambda within 1 SE 
  of the minimum, not the textbook largest/sparsest -- same 
  ceiling-drift failure mode as the grid cap, documented in the MATLAB 
  source itself). Diagnostic CV-curve plot dropped (UI-only, no bearing 
  on the return value), same as every other graphics() call already 
  dropped elsewhere in this port. 
  crossvalind('Kfold',...) (MATLAB Stats Toolbox) replaced with 
  sklearn.model_selection.KFold, per the existing decision already 
  recorded in this file's Katie conversion plan -- scikit-learn 
  installed 2026-07-29. Added an explicit `random_state` parameter (not 
  in the MATLAB signature) since MATLAB's fold assignment draws from 
  unseeded global RNG state with no equivalent to preserve. 
  TWO MORE CORRECTNESS TRAPS CAUGHT (same MATLAB/NumPy parity class as 
  build_sindy_library.py's/stlsq.py's ddof=1 issue): (1) `std(cv_error,0,2)` 
  needs ddof=1 again. (2) MATLAB's `max(lambda_floor, 0.005)` IGNORES NaN 
  (returns 0.005 when lambda_floor is NaN, e.g. every OLS coefficient is 
  exactly 0) -- NumPy's `max()`/`np.maximum` PROPAGATE NaN instead, which 
  would have silently corrupted the grid-trimming comparison. Used 
  `np.fmax` (NaN-ignoring, matches MATLAB here) instead. 
  test_cross_validate_lambda.py (3 tests): noisy synthetic data with a 
  real unprotected signal term + a protected tiny-magnitude term -- 
  verifies the selected lambda's full-data refit recovers exactly the 
  true support; an all-zero-dX case deterministically forces every fold 
  degenerate, verifying the RuntimeError path fires instead of silently 
  returning a meaningless lambda; and an end-to-end 
  build_sindy_library -> cross_validate_lambda -> stlsq composition 
  check at Katie's real dims.

- prune_sindy_bic.py (from prune_sindy_bic.m) — second, PER-EQUATION 
  refinement pass run after stlsq.py's single global lambda: for each of 
  the two dynamic states only (SCIM, BBS -- states 3-6 are static, 
  pruning them is meaningless), greedily removes the smallest-magnitude 
  active UNPROTECTED term one at a time, refitting after each removal, 
  accepting the removal only if BIC = N*log(RSS/N) + k*log(N) doesn't 
  increase by more than a 0.5-unit cushion. Answers a genuinely 
  different question than STLSQ's magnitude threshold: whether a term 
  that cleared the CV-selected lambda actually earns its keep once its 
  complexity cost is weighed against the fit it buys. `min(2, n_states)` 
  dynamic-states-only restriction and protected-term immunity are both 
  hardcoded SCI-domain assumptions carried over as-is from the MATLAB 
  source, not bugs. Same ddof=1 correctness trap as the three files 
  above (std(Theta,0,1)/std(dX,0,1) again need ddof=1). bic_history 
  returned as a list of dicts (one per pruned state) instead of a 
  MATLAB struct array -- nothing consumes it yet (would feed 
  plot_results.m, not yet ported). 
  test_prune_sindy_bic.py (4 tests): a permissively-thresholded STLSQ 
  fit that over-selected a noise-driven term gets it pruned while real 
  terms survive; a protected spurious-magnitude term survives BIC 
  pruning even though an equally-spurious unprotected term at the same 
  scale gets dropped; states 3-6 are confirmed byte-for-byte untouched 
  regardless of what's in them; and an end-to-end 
  build_sindy_library -> cross_validate_lambda -> stlsq -> 
  prune_sindy_bic composition check at Katie's real dims, confirming 
  pruning never increases nnz and protected action terms survive in 
  both dynamic-state equations.

- validate_sindy_model.py (from validate_sindy_model.m) — accept/reject 
  gate for the identified model (manuscript Section IV-B: accept if 
  NMAE < 1.0). Forward-simulates OPEN-LOOP/AUTOREGRESSIVE on held-out 
  patients (the model's own predicted previous state, not the true 
  observed state, feeds the next step) via Euler integration, x += 
  Theta(x,u) @ Xi clipped to [0,1] -- a materially harder test than 
  STLSQ/CV/BIC's one-step-ahead residual fitting, since small per-step 
  errors compound over a 12-week horizon. Scores only the two dynamic 
  states (SCIM, BBS) via NMAE (mean abs error / true data's own dynamic 
  range) and RMSE. Y_val and lib_labels are genuinely unused in the 
  MATLAB source itself -- kept for call-site signature fidelity, same 
  convention as Measurement.m's unused u/c. 
  LATENT BUG FOUND AND NOT REPLICATED: this file has its own THIRD 
  independent copy of the row-builder logic (build_sindy_library.m and 
  ContinuousStateDynamics.m are the other two -- see 
  continuous_state_dynamics.py's docstring). Its copy estimates state 
  count as `n_states_approx = length(z) - 3`, a leftover assumption 
  that n_actions is always 3 (from the older 3-control demo). At 
  Katie's real dims (length(z)=19) this happens to still give the 
  correct cubic-term count only because `min(n_states_approx, 2)` 
  clamps regardless -- for any other action count it would silently 
  misalign which library columns Xi's coefficients multiply against. 
  Renamed continuous_state_dynamics.py's `_build_sindy_row` to public 
  `build_sindy_row` and imported it here instead of copy-pasting a 
  third time on the Python side -- ContinuousStateDynamics.m's own copy 
  already uses the correct `length(x)`, so this reuses that (shared, 
  tested) implementation. Same "flag explicitly, don't silently 
  replicate a known-fragile formula" reasoning already applied to 
  backward_pass.py's Sxxh symmetry fix. 
  Also intentionally different from continuous_state_dynamics.py: no 
  `swat` recovery-ceiling clamp here, just a plain [0,1] box clip -- 
  this file evaluates the RAW identified model's predictive accuracy 
  against real data, independent of the controller's own dynamics 
  clamping (not an inconsistency). 
  test_validate_sindy_model.py (4 tests): a self-consistent perfect-model 
  case (X_val generated by literally running the same Euler rollout with 
  the same Xi) gives ~0 NMAE/RMSE and bit-exact X_sim; an all-zero-Xi 
  case (model predicts no change) gives hand-computed NMAE/RMSE matching 
  by hand; engineered static-state "drift" in the ground truth is 
  confirmed invisible to scoring (dynamic-states-only restriction); and 
  an end-to-end build_sindy_library -> cross_validate_lambda -> stlsq -> 
  prune_sindy_bic -> validate_sindy_model composition check on a 
  held-out split at Katie's real dims.

- compute_fisher_information.py (from compute_fisher_information.m) — 
  6th of 7 files in the "SINDy" list (build_sindy_library, stlsq, 
  cross_validate_lambda, prune_sindy_bic, validate_sindy_model, 
  compute_fisher_information, print_identified_equations) -- only 
  print_identified_equations.py remains from that list after this. 
  Computes F = (1/sigma^2) * Theta_n.T @ Theta_n, the closed-form Fisher 
  information matrix for a linear-Gaussian model -- not learned from 
  data, a property of the design (which library terms, how much 
  training variation). Its inverse is the Cramer-Rao lower bound on 
  achievable SINDy coefficient uncertainty; the dual iLQG controller 
  uses it to quantify expected information gain from a proposed action 
  (curiosity = prefer actions that shrink parameter uncertainty 
  fastest), connecting the batch identification pipeline to the online 
  dual-control loop (todorov_estimator.py/spkf_function.py). 
  Same ddof=1 correctness trap as every file in this pipeline 
  (std(Theta,0,1) needs ddof=1). MATLAB's two `warning(...)` calls 
  (near-zero noise_std fallback to 1e-6; non-PSD F regularisation) are 
  translated to Python's `warnings.warn` rather than dropped -- both are 
  reachable with realistic inputs and meant to surface as diagnostics, 
  unlike the dead/unreachable warnings dropped elsewhere in this port. 
  `eig(F)` -> `np.linalg.eigvalsh(F)` (F is symmetrised immediately 
  before, so this is the natural equivalent, not a deviation). 
  test_compute_fisher_information.py (5 tests): matches the closed-form 
  formula directly; near-zero noise_std triggers the documented 1e-6 
  fallback WITH a captured warning; F is symmetric/PSD for random data; 
  F is invariant to arbitrary per-column physical-unit rescaling of 
  Theta (the whole point of normalising internally -- a real correctness 
  property, not just a sanity check); and an end-to-end composition 
  check at Katie's real dims that INCIDENTALLY (and now explicitly 
  asserted on) exercises the not-PSD regularisation branch for real: 
  poly_order=2+cross with only 20 patients builds MORE library terms 
  (210) than training observations (200), so Theta.T @ Theta is 
  genuinely rank-deficient, not just poorly conditioned -- exactly the 
  scenario that branch exists for.

- print_identified_equations.py (from print_identified_equations.m) — 
  LAST file of the 7-file "SINDy" list (build_sindy_library through 
  print_identified_equations) -- SINDy pipeline now fully ported. Not 
  math -- the clinician-interpretability layer (manuscript Section 
  IV-E): prints the sparse ODEs with physically meaningful variable 
  names and signed coefficients (SCIM/BBS equations only; static states 
  already zeroed), plus a short "clinical interpretation" line 
  translating the top-2 dominant terms per equation into plain language 
  (a7->U7, x1->SCIM, ...) and increases/decreases direction, and an 
  overall sparsity summary. `cfg` unused in the MATLAB source itself 
  despite being documented as "(.verbose)" -- kept for signature 
  fidelity, same convention as Measurement.m's unused u/c. 
  COMMENT/CODE MISMATCH FOUND (flagged, not corrected): the MATLAB 
  source's comment above the a(\d+)->U$1 substitution claims a "word 
  boundary so that a1 does not partially match a10..a13" -- but the 
  actual regex has no boundary anchor there (unlike the x1..x6 
  substitutions, which do use one). Still correct in practice, just not 
  for the stated reason: `\d+` is greedy, so a(\d+) applied to "a10" 
  matches the whole "a10" in one shot -- no boundary was ever needed for 
  this pattern. Ported literally (no boundary on a(\d+), matching the 
  x1..x6 substitutions' MATLAB `\>` with Python `\b`). Two DIFFERENT 
  "is this zero" checks preserved as-is rather than unified: 
  abs(xi_s)>1e-8 gates which terms get PRINTED per equation, but the 
  final sparsity summary uses exact np.count_nonzero -- same 
  distinction the MATLAB source itself makes. 
  Minor testability addition beyond the literal MATLAB signature (no 
  return value there): returns the full report string in addition to 
  printing it, so tests can assert on content without capturing stdout. 
  test_print_identified_equations.py (5 tests): equation formatting + 
  scale-tag stripping; clinical interpretation only covers the top-2 
  terms by magnitude (a 3rd, smaller term is confirmed to appear in the 
  equation but NOT the interpretation bracket); a1 vs a10 is confirmed 
  not to collide (U1/U10 both render correctly, no "U100" leak); the 
  two-different-"nonzero"-checks distinction is directly exercised with 
  a 1e-9 coefficient (excluded from the printed equation, still counted 
  in the sparsity summary); and an end-to-end 6-stage pipeline 
  composition check at Katie's real dims produces a genuinely readable 
  report (21/210 active SCIM terms, 18/210 active BBS terms).

## STATUS — Zahid conversion: IN PROGRESS (started 2026-08-04)
Python code was originally written in PYTHON/zahid/; as of the 
2026-08-17 reorganization it's split between PYTHON/extensions/
constraints/ (dynamic_control_bounds.py, relative_degree_reduction.py) 
and PYTHON/applications/pendulum_constrained/ (run_zahid_demo.py, 
pendulum_demo_chart.png) — see "Repository layout" above. Zahed 
Dastan's actual codebase was 
never received and no public repo exists for it (checked: no URL/GitHub 
link anywhere in his paper, no data-availability section, no findable 
public repo under his name -- confirmed 2026-08-04). Draft email sent 
asking him directly. UNLIKE every other conversion in this project 
(Todorov/Andrew/Katie), this is NOT a MATLAB-to-Python port -- there is 
no source to diff against. It's an ORIGINAL implementation derived 
directly from the equations in his paper: Dastan, Z. & Sensinger, J. 
(2024), "Leveraging Control Inputs to Enforce Constraints in 
Differential Dynamic Programming for Nonlinear Optimization," IEEE 63rd 
CDC, DOI 10.1109/CDC56724.2024.10886170 (PDF: CODES/ZAHID'S PAPER.pdf). 
Verification is against the paper's own worked examples (hand-derived 
closed forms, qualitative match to reported figures), not bit-exact 
numeric parity -- a real, flagged limitation of this branch relative to 
the other three.

Method summary: extends Tassa/Mansard/Todorov's control-limited DDP 
(same base as core/ddp_solver/) to handle general inequality constraints 
h(x,u)>=0 or h(x)>=0, WITHOUT touching the Q/V-function recursion at 
all (backward_pass.py-equivalent math is untouched) -- the only change 
is that the box bounds fed into the existing box-QP solver 
(box_qp.py-equivalent) are recomputed every outer iteration from the 
CURRENT nominal trajectory instead of being one fixed constant for the 
whole run. Two variants: (A) h(x,u)>=0 directly gives state-dependent 
bounds; (B) h(x)>=0 (state-only) is first reduced to a control-dependent 
form via one Lie derivative + a control-barrier-function-style 
alpha*h(x) decay term (eq 16-19: h_tilde = dh/dx . f(x,u) + alpha*h(x)), 
then handled by the same Variant-A machinery. Would extend 
core/ddp_solver/back_pass.py and core/ddp_solver/forward_pass.py (small, 
backward-compatible: `lims` gains an optional per-timestep-array shape 
alongside its current constant-(m,2) shape) and core/ddp_solver/ilqg.py 
(new optional `constraint_fn` parameter, recomputed once per outer 
iteration); core/ddp_solver/box_qp.py needs zero changes -- it already 
accepts arbitrary bounds per call.

**dynamic_control_bounds.py (Variant A, eq 10-13): DONE.** 
`solve_box_from_constraint(h_fn, x_i, u_i, dhdu_fn=None, lower=-inf, 
upper=inf)` -- first-order-solves each constraint row for the control 
channel its Jacobian is nonzero in, deriving an upper or lower bound 
depending on the Jacobian's sign, intersected with any pre-existing hard 
limits. Falls back to a numeric Jacobian via extensions/dual_control/
finite_difference.py (reused directly; as of the 2026-08-17 
reorganization this is a package import, `from 
extensions.dual_control.finite_difference import finite_difference`, 
not the sys.path hack originally used here) when no analytic `dhdu_fn` 
is given. 
Raises ValueError (not a silent wrong answer) if a constraint row's 
Jacobian has more than one nonzero control-channel entry -- inherited 
restriction from the paper itself (every one of its worked examples has 
relative degree one w.r.t. exactly one channel; a box bound can't 
express a multi-channel half-space). Warns (doesn't silently drop) when 
a row's Jacobian is entirely zero w.r.t. u at a given point -- a real 
diagnostic that relative-degree-one doesn't hold there. 
`build_time_varying_lims(h_fn, x_traj, u_traj, ...)` loops this over a 
whole nominal trajectory, producing the (N, m, 2) array 
back_pass.py/forward_pass.py's planned extension will consume 
(x_traj/u_traj follow ilqg.py's existing x[:, :N]/u convention -- N 
control steps, not N+1). 
VERIFICATION (test_dynamic_control_bounds.py, 12 tests): hand-derived 
the paper's own inverted-pendulum example (Sec IV-A, eq 20-23) by hand 
-- h(x)=1-omega (constraint omega<1), continuous dynamics 
phi_dot=omega/omega_dot=sin(phi)+u implied by the paper's Euler-discretized 
eq (20), giving h_tilde(x,u) = alpha*(1-omega) - sin(phi) - u via eq 19 
(alpha=0.1) -- and confirmed the module reproduces the closed-form 
u_max = alpha*(1-omega) - sin(phi) exactly (to ~1e-10) at 4 trajectory 
points, via BOTH the analytic-Jacobian path and the finite-difference 
path (the latter matches to ~1e-6 rather than 1e-10 only because of 
forward-difference float roundoff -- h_tilde is exactly affine in u so 
there's zero truncation error, a strong test since any real bug would 
blow well past that tolerance). Also tested: hard-limit intersection 
(both "hard limit inside constraint bound" and "hard limit tighter than 
constraint bound" cases), the multi-channel-Jacobian ValueError path, 
the zero-Jacobian RuntimeWarning path, a vector h with two independent 
scalar rows each correctly pinning a different side of the box, and 
build_time_varying_lims end-to-end over a 3-step synthetic trajectory.

**relative_degree_reduction.py (Variant B, eq 15-19): DONE.** 
`lie_derivative_h(h_fn, dynamics_fn, x, u, dhdx_fn=None)` computes 
h_dot(x) = dh/dx . f(x,u) (eq 16-17) -- NOTE dynamics_fn here must be the 
CONTINUOUS-time vector field x_dot=f(x,u) (paper eq 1), not a discrete 
one-step map like ilqg.py's step_fn; the chain rule in eq 16 is a 
continuous-time construction. Falls back to a numeric dh/dx via 
extensions/dual_control/finite_difference.py when no analytic dhdx_fn 
is given (same reuse pattern as dynamic_control_bounds.py). 
`state_constraint_to_control_constraint(h_fn, dynamics_fn, alpha, 
dhdx_fn=None)` returns a new callable h_tilde(x,u) = h_dot(x) + 
alpha*h(x) (eq 18-19) -- a drop-in `h_fn` for 
dynamic_control_bounds.py's Variant-A functions, exactly matching the 
paper's own description of the reduction ("directly apply the 
methodology developed in the previous subsection"). alpha is a required 
argument, not defaulted -- the paper gives no selection rule and uses 
different values per constraint even within one experiment (0.1 for the 
pendulum; 0.2/0.01/0.02 for the 2D car's three separate obstacles). 
VERIFICATION (test_relative_degree_reduction.py, 19 tests): reproduced 
the SAME pendulum closed form as dynamic_control_bounds.py's tests 
(h_tilde = 0.1*(1-omega) - sin(phi) - u), but this time generated 
AUTOMATICALLY from nothing but h(x)=1-omega and the raw continuous 
dynamics phi_dot=omega/omega_dot=sin(phi)+u, instead of hand-derived -- 
exact match (~1e-10) via the analytic-Jacobian path, both here (dh/dx) 
and in dynamic_control_bounds.py (dh_tilde/du), since h(x)=1-omega is 
affine in x so finite-differencing it has zero truncation error either 
way. Includes explicit END-TO-END COMPOSITION tests: feed the 
auto-generated h_tilde_fn straight into 
dynamic_control_bounds.solve_box_from_constraint and confirm the 
resulting bound reproduces the identical closed-form u_max as 
dynamic_control_bounds.py's own standalone tests -- including a 
fully-numeric run (no analytic Jacobian anywhere in either module, both 
falling back to finite differences) landing within 1e-4. Also tested: 
alpha<=0 raises; a genuinely relative-degree-two constraint (Lie 
derivative provably independent of u) is confirmed independent of u 
directly, AND confirmed to trigger dynamic_control_bounds.py's 
RuntimeWarning when composed downstream; and a vector h with one row 
that depends on u and one that doesn't, at real dims, matching both 
rows' closed forms by hand.

**core/ddp_solver/back_pass.py, core/ddp_solver/forward_pass.py, 
core/ddp_solver/ilqg.py extensions: DONE (2026-08-04, paths updated 
2026-08-17 reorganization).** Small, additive, backward-compatible --
existing (m,2)-shaped `lims` call sites are untouched.
- back_pass.py: `lims` may now be (m,2) [original] or (N-1,m,2) 
  [per-timestep, Dastan & Sensinger]. The `lims[0,0]>lims[0,1]` 
  "unconstrained" sentinel now only applies when `lims.ndim==2` (it 
  never meant anything for a per-timestep array). Inside the loop, 
  `lims_i = lims if lims.ndim==2 else lims[i]` selects the row for the 
  current timestep before building the box-QP bounds -- this one line is 
  the entire mechanism by which the paper's state-dependent bound 
  reaches the solver.
- forward_pass.py: identical `lims.ndim` branch around its clip line, so 
  the line-search rollout clips each candidate to the SAME per-timestep 
  bound used by that iteration's backward pass (computed from the 
  reference trajectory, not recomputed per candidate alpha -- matches 
  the paper's own Fig. 2 / eq 14 "modest step" argument).
- ilqg.py: new optional `constraint_fn(x_traj, u_traj) -> (N,m,2)` 
  parameter. Recomputed once per outer iteration, gated by the same 
  `flg_change` flag that already gates re-differentiating the dynamics 
  (no point recomputing bounds around a trajectory that wasn't 
  accepted) -- `lims = constraint_fn(x[:, :N], u)` right after the 
  derivs_fn() call, before back_pass() is called. Nothing else in the 
  loop changes: back_pass()/forward_pass() already just read whatever 
  `lims` currently holds. The initial divergence-avoiding rollout (before 
  the main loop) intentionally does NOT apply constraint_fn -- documented 
  as intentional, since it only needs *a* non-diverging trajectory to 
  seed iteration 1; real constraint enforcement starts at the first 
  backward pass. box_qp.py needed ZERO changes (confirms the earlier 
  analysis -- it already solves an arbitrary box per call).
REGRESSION CHECK: all pre-existing tests pass unchanged, run the way 
this project has always tested at the time (per-directory, not as one 
combined top-level pytest invocation -- todorov/forward_pass.py and 
andrew/forward_pass.py shared a bare module name with completely 
different signatures, and running the whole PYTHON/ tree in one pytest 
process hit a pre-existing, order-dependent sys.modules collision 
between them; confirmed this collision existed independent of that 
day's changes by reproducing it with only andrew+todorov present, in 
both collection orders -- not a regression, a pre-existing structural 
fact of the repo's flat same-named-modules-per-directory layout, out of 
scope at the time). Per-directory: todorov 6/6, andrew 16/16, katie 
79/79, zahid 31/31 -- 132/132, zero regressions. RESOLVED by the 
2026-08-17 core/extensions/applications reorganization (see "Repository 
layout" above): every module now has a fully-qualified package name, so 
the collision is gone and the entire PYTHON/ tree runs as a single 
`pytest` invocation -- reconfirmed 132/132 passing that way post-move.

**run_zahid_demo.py: DONE (2026-08-04).** Reproduces the paper's own 
inverted-pendulum experiment (Sec IV-A, eq 20-23) end-to-end through the 
REAL ilqg(), with constraint_fn built by composing 
relative_degree_reduction.state_constraint_to_control_constraint (omega<1 
-> h_tilde) with dynamic_control_bounds.build_time_varying_lims 
(h_tilde -> per-timestep box) -- the actual composition the paper 
describes for a pure state constraint.

FINDING (worth telling supervisor, same spirit as the backward_pass.m 
symmetry bug found earlier in this project): the paper's LITERAL 
alpha=0.1 TRAPS this DDP implementation after exactly one accepted outer 
iteration (11 iterations total, exits via lambda > lambda_max, final 
cost stuck at 149.6 vs the paper's reported 10.11). Confirmed this is 
NOT a coding bug, not a floating-point fluke at the exact phi=-pi point, 
and not fixable by naive optimizer-parameter tweaking -- root cause 
diagnosed by direct inspection: u_max(x) = alpha*(1-omega) - sin(phi) 
stays in a genuinely narrow ~0.1-0.25 window across roughly phi in 
[-pi, -3.0] (about the first 0.14 rad of the swing) BECAUSE alpha=0.1 is 
conservative even where h=1-omega is maximally far from its boundary 
(h=1 at x0) -- a real property of the eq 18 CBF-style condition at small 
alpha, not specific to the exact pi value. The backward pass's box-QP 
has essentially no room in that stretch; after iteration 1 exhausts what 
little slack exists, every later iteration is rejected and lambda climbs 
past lambda_max. Verified the unconstrained version of this SAME 
step_fn/derivs_fn converges cleanly (cost ~8.9 in 7 iterations, 
max|omega| naturally reaches 1.42 -- confirms the constraint is 
genuinely binding, not vacuous, and confirms step_fn/derivs_fn 
themselves are correct).

This directly instantiates a limitation the paper's own eq 14 states 
("as long as the adjustments to u are MODEST... the updated trajectory 
will continue to fulfill the inequality constraint") -- confirmed 
empirically via an alpha sweep with the SAME unmodified code: alpha=0.3 
still traps AND the returned trajectory actually VIOLATES the constraint 
(max omega ~1.52, from one large accepted step overshooting the box that 
was linearized around the old reference); alpha=2.0 undershoots the trap 
less badly but still slightly violates (~1.05, cost 9.87 -- very close 
to the paper's 10.11); alpha=1.0 converges CLEANLY with the same code 
(13 iterations, cost 11.46, max omega 0.925 -- stays under 1) -- a 
genuine, honest qualitative match to Fig 3 (paper: 16 iterations, cost 
10.11, "omega stays close to but under 1"). The demo runs alpha=0.1 
first and reports the trap plus its diagnosis explicitly, THEN alpha=1.0 
as the actual qualitative-match result, rather than silently 
substituting one for the other -- alpha has no closed-form selection 
rule in the paper itself (confirmed already in 
relative_degree_reduction.py's docstring: the paper uses 3 DIFFERENT 
alpha values even within its own single 2D-car experiment), so choosing 
a different one to demonstrate correctness is a disclosed deviation, not 
a silently swapped parameter. 
Chart (PYTHON/applications/pendulum_constrained/pendulum_demo_chart.png, 
alpha=1.0 run) is a strong 
qualitative match to paper Fig 3a/3b -- omega rises smoothly toward but 
never crosses the omega=1 line, then decays to 0; cost decreases 
monotonically to convergence.

Zahid conversion is now feature-complete relative to the plan: both 
Variant A (dynamic_control_bounds.py) and Variant B 
(relative_degree_reduction.py) implemented and unit-tested against 
hand-derived closed forms, wired into the real solver 
(back_pass.py/forward_pass.py/ilqg.py extensions, zero regressions on 
132 pre-existing tests), and composed end-to-end on the paper's own 
worked example with an honestly-reported, root-cause-diagnosed 
qualitative match.

## STATUS — linear dual-control example (thesis Sec. 3.4): DONE (2026-08-26)
Reproduces thesis Figs. 3.4/3.5 through the REAL 
extensions/dual_control/main_outer_control_loop.py MPC loop (no solver 
changes) -- the only new code is the plant callables themselves, in 
PYTHON/applications/linear_dual_control/ (linear_plant.py + 
run_linear_demo.py; see "Repository layout" above). This is Andrew's 
first applications/-level plant (previously extensions/dual_control/ 
had no applications/ entry at all -- see "Repository layout").

Plant: x1_dot = d1*u1 - d2, x2_dot = d2*u2, d=[d1,d2] unknown/estimated 
(augmented onto the state, xa=[x1,x2,d1,d2]); cost (x1-5)^2 + 
1e-4*u2^2. x0=[0,0], d_true=[125,50], d_hat_0=[100,30], 
Sigma_d0=[[8100,4500],[4500,5625]], 6 steps x dt=0.05s, measurement 
noise G=1e-2, process noise F=1e-15 for BOTH state and parameter 
dynamics. Verified directly against the thesis PDF (NOTES/2024_Mathis_
Sensinger Carretero_PhD Thesis.pdf, pp. 58-61) on 2026-08-26 -- states 
AND controls are both explicitly stated unconstrained (no u1/u2 bound 
anywhere in Sec. 3.4); lambda_i/reg_type/horizon_mode/cov_X are not 
stated for this experiment anywhere in the thesis (checked Sec. 2.1.3's 
generic iLQG background and Appendix A's DDP-comparison tables too) -- 
left at main_outer_control_loop.py's defaults. Validated against 
extensions/dual_control/protocols.py's DynamicsFn/MeasurementFn/CostFn/
ContinuousDynamicsFn contracts via validate_plugin.py before running 
(augment_states=True only -- this problem always augments d1/d2, the 
False branch was never exercised).

RESULT (resolved 2026-08-26, see below): total cumulative cost = 0.391, 
matching the thesis's reported 0.4. d_hat converges to within ~0.03% of 
the true [125,50] after just the FIRST session (thesis: "accurately 
estimating the true parameter values... after the first time step"); x1 
overshoots to 5.625 on session 1 (thesis: "overshot to 5.6") then tracks 
5 with negligible error afterward (thesis: "tracked the state reference 
with only small errors"); u2 shows genuine dual-control probing, 
decaying from ~0.3 to ~0 over about four sessions (thesis: "gradually 
steps down to zero over four time steps"); virtually the entire 0.391 
traces to the single first-step overshoot, (5.625-5)^2=0.3906. All four 
of the thesis's own qualitative claims for this figure now match.

FIRST DIAGNOSIS WAS WRONG, CORRECTED SAME SESSION: an earlier pass 
reported 1.128 (2.8x too high) and hypothesized an unstated control 
limit or solver-default mismatch. Re-reading the thesis PDF directly 
(not just the headline summary this task started from) disproved that: 
no control limits, and the thesis's own numbers (5.6 overshoot) already 
matched the 1.128 run almost exactly. The REAL cause was 
main_outer_control_loop.py's own documented "NOT PORTED: multi-seed 
first-iteration search" limitation, initially ruled out incorrectly -- 
the first warm-start check only varied u1's seed, which turned out not 
to be the sensitive dimension. All-zero u_bar_0 (the library's own 
default) lands the first iLQG solve in a real local optimum where u2 
stays ~1e-6 the entire horizon (no probing at all, despite Sigma_d0 
being huge) -- confirmed genuinely LOCAL, not global, by re-testing with 
a nonzero u2 seed: ANY meaningfully nonzero u2 warm-start (tested 0.05 
to 0.5, both signs, flat or decaying profiles) escapes it and lands 
within noise of 0.391. run_linear_demo.py's `run()` now reproduces 
thesis Sec. 3.3.1's own stated seeding practice by hand (re-running the 
whole outer loop from a small set of u_bar_0 candidates for the FIRST 
solve only, keeping the lowest-cost result) since main_outer_control_loop's 
`first_run_seeds>1` itself is NotImplementedError.
Chart: PYTHON/applications/linear_dual_control/linear_demo_chart.png.

KNOWN GAP, DELIBERATELY NOT GENERALIZED (discussed with supervisor 
2026-08-26): main_outer_control_loop.py's `first_run_seeds>1` remains 
NotImplementedError -- proper multi-seed support (matching the thesis's 
Sec. 3.3.1 method, e.g. random N(0, 0.01) initial control sequences per 
Sec. 3.5's later description) was deliberately NOT built now. This one 
linear-example case is the only place in the codebase that has hit a 
bad-local-optimum result from the all-zero u_bar_0 default; building a 
generic seeding API from a single data point risks guessing wrong about 
what actually needs to vary (u_bar_0 only? random draws vs. a small 
fixed candidate set like run_linear_demo.py's? seeded every outer step 
vs. just the first, per Sec. 3.3.1's own stated tradeoff?). Revisit and 
build it properly into main_outer_control_loop.py if a SECOND real case 
hits the same issue -- most likely candidate: applications/sci_rehab/
run_katie_pipeline.py's dual-iLQG control stage, which also warm-starts 
u_bar from zero/from the previous session's shift, same as this example 
did before seeding was added by hand.

ADDENDUM (2026-08-27): applications/duopoly_pricing/ (see "STATUS — 
duopoly pricing" below) DID test multi-seed warm-starting against a 
second real dual-iLQG application, but did NOT confirm this gap -- its 
under-identified cross-price parameters turned out to be a genuinely 
DIFFERENT phenomenon (lack of persistent excitation once price settles, 
not a bad local optimum: several very different warm starts all 
converged to the identical trajectory there). So the "still only one 
confirmed case" status of this local-optimum gap is unchanged; Katie's 
pipeline remains the best candidate to actually confirm or retire it.

## STATUS — duopoly pricing (applications/duopoly_pricing/): DONE (2026-08-27)
Two-seller generalization of applications/dynamic_pricing/ (see 
"Repository layout" above for the backfilled note on that folder). ONE 
joint controller sets both sellers' prices to maximize COMBINED profit 
(explicit framing choice, confirmed with supervisor during math review 
before coding -- this codebase implements optimal control, not game 
theory, so "competition" is modeled as cross-price terms coupling two 
demand curves, not a Nash equilibrium between independently-optimizing 
agents). No changes to core/ddp_solver/protocols.py or 
extensions/dual_control/protocols.py, per explicit instruction -- 
confirmed by re-running the full PYTHON/ suite (134/134, zero 
regressions) and by the fact validate_dual_control_plugin() (unmodified) 
accepted this plug-in at nx=2/nu=2/n_p=4/ny=2/nv=2 under BOTH 
augment_states modes, not just True (unlike linear_plant.py, which only 
supports True -- see continuous_dynamics.py's augment_states=False 
branch, matching dynamic_pricing/continuous_dynamics.py's pattern).

Math: D_A(p_A,p_B;b) = a_A - b_AA*p_A + b_AB*p_B, D_B symmetric -- 4 
INDEPENDENT unknown sensitivities [b_AA, b_AB, b_BA, b_BB] (own + cross 
per seller, not a symmetry-constrained 3), all 8 other economic knobs 
(a, c, eta, p_ref, F_dyn, sigma_y, lambda_reg) per-seller rather than 
shared -- both per explicit instruction from the math-review discussion. 
Applied dynamic_pricing/'s two hard-won lessons from the start rather 
than rediscovering them: cost.py prices off the CURRENT price-implied 
demand (not the lagged state), and u_lim_method=1/box-QP bounds (not 
u_lim_method=2's tanh-squash).

TWO REAL FINDINGS from actually running the demo (not hypothetical, same 
"run it end-to-end, don't just validate shapes" standard as every other 
STATUS entry in this file):

(1) FOUND AND FIXED: initial constants (cross-terms b_AB=0.8, b_BA=0.6) 
put the TRUE unconstrained joint-profit optimum at (p_A*,p_B*)=(45.4, 
49.2) -- AT/PAST P_MAX=45, so price stayed permanently pinned at the 
ceiling even once the estimate was fully informed. This is a real 
economic effect, not a bug: a substitute-goods duopoly under single joint 
ownership has a genuine incentive to price ABOVE the competitive level 
(raising p_A pushes customers to B, a positive externality the joint 
owner internalizes) -- confirmed by solving the FOC analytically. With 
price pinned CONSTANT, D_A=a_A-b_AA*p+b_AB*p collapses to a function of 
only the DIFFERENCE (b_AA-b_AB); the pinned run's b_hat converged to 
[0.95,-0.25,-0.25,0.95] -- sign-flipped and wrong on every individual 
component, yet (b_AA_hat-b_AB_hat)=(b_BB_hat-b_BA_hat)=1.2 EXACTLY 
matched the true differences (2.0-0.8=1.2, 1.8-0.6=1.2). FIXED by 
choosing milder cross-terms (0.3/0.25) that keep the true optimum 
interior (34.5, 38.2), well clear of both P_MAX and the choke prices.

(2) FOUND, NOT FIXED, HONESTLY REPORTED INSTEAD (see run_duopoly_demo.py's 
module docstring and print_report()'s output): even with an interior 
optimum, prices settle near a fixed point by session ~2 and barely move 
afterward, so only each seller's IDENTIFIABLE combination (b_AA-b_AB, 
b_BB-b_BA) converges well (2-3% error) -- individual b_AA/b_BB (own-price) 
improve substantially but don't fully converge, and individual b_AB/b_BA 
(cross-price) do NOT reliably improve at all, ending up numerically wrong 
in the reference run. VERIFIED this is NOT a local-optimum artifact 
(ruling out the linear_dual_control precedent's fix): re-running the 
first solve from three very different warm-start seeds (oscillating, 
ramping-apart, random price sequences) converged to the bit-identical 
trajectory every time. Diagnosed instead as a textbook lack of 
PERSISTENT EXCITATION -- with price essentially constant, D_A is observed 
at one repeated (p_A,p_B) pair, which identifies only the one linear 
combination governing D_A, not the two coefficients separately. Root 
cause is a genuine qualitative difference from the linear thesis example: 
there, u2 was nearly free (1e-4 weight) so active probing was cheap and 
worthwhile; HERE, both actions are real prices with direct profit 
consequences, so the dual controller correctly declines to sacrifice real 
profit purely to resolve an ambiguity that doesn't change its current 
pricing decision -- a rational, not broken, response given the cost as 
specified. run_duopoly_demo.py's docstring and print_report() were 
rewritten to claim only what's actually true (combination convergence, 
not full individual-component convergence) rather than ship a demo whose 
own stated success criteria its own output would contradict.

Files: PYTHON/applications/duopoly_pricing/duopoly_constants.py, 
continuous_dynamics.py, dynamics.py, measurement.py, cost.py, 
test_plugin_contract.py, run_duopoly_demo.py.

## STATUS — 2-link robot arm (applications/robot_arm_2link/): DONE (2026-08-30)
Math/mapping proposed and reviewed BEFORE any code was written (per 
explicit instruction), then built in the same order the user specified: 
constants object -> continuous_dynamics -> dynamics -> measurement -> 
cost -> validate -> run. First genuinely robotics-framed plug-in in this 
codebase (every prior application/ plant is economic or biomedical) and 
the first with a real coupled nonlinear ODE (inertia-matrix inversion, 
Coriolis terms, gravity) rather than a linear/near-linear plant.

Math (standard rigid-body manipulator equation, Spong/Hutchinson/
Vidyasagar; unknown-payload-mass benchmark per Slotine & Li): 2-DOF 
revolute-revolute arm in a vertical plane (gravity ON, per explicit 
choice over the horizontal/no-gravity alternative). States (nx=4) 
x=[q1,q2,w1,w2] (joint angles + angular velocities); actions (nu=2) 
u=[tau1,tau2] (joint torques); ONE unknown parameter (n_p=1) m2 = link 
2's mass, chosen over a friction coefficient because it multiplies 
EVERY term of M(q)/C(q,qdot)/G(q), not just one. 
M(q)@qddot + C(q,qdot)@qdot + G(q) + Fv@qdot = tau, Fv=diag(fv1,fv2) 
known viscous friction; qddot solved via a closed-form 2x2 matrix 
inverse (vectorized across the (nxa,K) batch shape every plant in this 
codebase uses -- no per-column linalg.solve loop), confirmed invertible 
even at m2->0 (det = I2*(m1*lc1^2+I1) > 0) since I2>0, so no special- 
casing was needed for validate_plugin.py's own uniform-[0,1] synthetic 
sampling (which does sample m2 near zero). Joint-space target (q1_target, 
q2_target), NOT Cartesian end-effector -- explicit choice, keeps this 
first version to one source of nonlinearity (the manipulator dynamics 
itself) rather than adding forward-kinematics on top.

Cost: quadratic joint-angle tracking + velocity penalty (keeps the 
trajectory from swinging wildly en route) + torque-effort penalty, with 
a terminal weight boost (20x running weight) so "reach the target" means 
settle there at rest -- same running-vs-terminal-boost convention as 
applications/sci_rehab/l_cost.py, terminal cost ADDS to running cost 
rather than replacing it (also matching l_cost.py, not 
duopoly_pricing/cost.py's zero-terminal convention).

Illustrative textbook-typical numeric values (per explicit instruction 
to propose them): l1=l2=1.0 m, lc1=lc2=0.5 m, m1=1.0 kg (known), 
I1=I2=0.1 kg*m^2, g=9.81, fv1=fv2=0.5 N*m*s/rad. m2_true=1.0 kg, 
m2_prior=0.3 kg (deliberately wrong, badly understating link 2's mass -- 
same "deliberately wrong initial guess" convention as every other dual- 
control demo in this codebase). Target q1=60 deg, q2=-45 deg from an 
at-rest horizontal start (q=w=0). Torque bounds tau1 in [-30,30] N*m, 
tau2 in [-15,15] N*m, u_lim_method=1 (box-QP, per the lesson already 
learned and written up under dynamic_pricing/duopoly_pricing rather than 
rediscovering it) -- checked directly against the TRUE arm's worst-case 
static gravity hold (q1=q2=0, both links fully extended: G1=19.62 N*m, 
G2=4.905 N*m), leaving ~10 N*m headroom on each joint for acceleration. 
40 sessions x dt=0.05 s (2.0 s reach horizon).

VALIDATION: extensions/dual_control/validate_plugin.py's 
validate_dual_control_plugin passes at nx=4/nu=2/n_p=1/nv=4/ny=4 under 
both augment_states modes. Full PYTHON/ suite: 262/262, zero regressions 
(261 pre-existing + this folder's one new contract test).

RESULT, run end-to-end via run_robot_arm_demo.py (real numbers, not 
hypothetical -- same "run it, don't just assert it" standard as every 
other STATUS entry in this file): 
(1) PERSISTENT EXCITATION PREDICTION CONFIRMED: unlike duopoly_pricing's 
price (settles to a fixed point and stops exciting the unknown once 
profit-optimal -- see that STATUS entry), reaching q1_target/q2_target 
from rest structurally requires the arm to move through a range of q2, 
which excites m2's effect on the inertia/Coriolis/gravity terms with NO 
dedicated probing incentive needed. m2_hat converged from the wrong 
prior (0.3 kg, |error|=0.700) to 1.017 kg (|error|=0.017) by session 
~9-10, purely from the reaching transient. 
(2) Target reached closely: final q1=65.66 deg (target 60, error 5.66 
deg), q2=-43.92 deg (target -45, error 1.08 deg), residual velocity 
~0.1 rad/s (near rest) -- a real but modest steady-state offset, not 
exact convergence, plausibly from the fixed-length (not shrinking) 
receding horizon (horizon_mode=2 default) not tightening urgency near 
the end; not investigated further as a tuning question, since the 
qualitative behavior (both dual control AND reaching) is what this demo 
was built to confirm. 
(3) GENUINE, EXPLICABLE TRANSIENT, not a bug: q2 overshoots hard early 
(to -101.9 deg at session 11, well past the -45 deg target) WHILE m2_hat 
is still far below truth (sessions 0-9), then corrects back to settle 
near target as the estimate converges (~session 10 onward) -- the wrong- 
mass belief drives wrong momentum/braking calculations during the 
transient, self-corrected once identification catches up. This is dual 
control's core mechanism (act under uncertainty, learn, correct) 
directly visible in the trajectory, reported as a genuine finding rather 
than tuned away. Torque stayed within bounds throughout (max 
29.17/13.75 N*m against the 30/15 limits) -- no saturation-trapping, 
unlike dynamic_pricing's original u_lim_method=2 failure mode.

Files: PYTHON/applications/robot_arm_2link/arm_constants.py, 
continuous_dynamics.py, dynamics.py, measurement.py, cost.py, 
test_plugin_contract.py, run_robot_arm_demo.py.

## STATUS — model-building wizard (PYTHON/wizard/): DONE (2026-08-27)
Guided, question-based Streamlit interface so someone can define and 
test a NEW model without writing raw Python -- explicitly domain-generic 
(never references SCI/pricing/duopoly-specific vocabulary in its own 
question wording, generated code, or logic), design proven against BOTH 
existing plug-in examples (dynamic_pricing/'s 1-state/1-action/1-unknown 
case AND duopoly_pricing/'s 2-state/2-action/4-unknown case with 
cross-terms) before being trusted generically. Full plan proposed and 
approved before building (question flow, file/config structure, 
equation-parsing safety design, error-handling design, Streamlit 
reliability design, documentation recommendation) -- see chat history 
for the complete proposal; summary of what shipped below.

**Zero changes to core/ddp_solver/protocols.py or 
extensions/dual_control/protocols.py** -- confirmed by construction (the 
wizard only calls the existing validate_dual_control_plugin() and 
main_outer_control_loop(), never edits either package) and by the full 
PYTHON/ test suite passing 261/261 (up from 134 pre-wizard) with zero 
regressions.

**Pipeline, all independently unit-tested (schema.py/equation_parser.py 
first, as the safety-critical pieces, per explicit instruction, THEN 
generic_plant.py, THEN the Streamlit question flow -- same order 
approved in the plan):**
- schema.py -- generic ModelConfig dataclasses (states/actions/unknown 
  parameters/constants/dynamics/measurement/cost/solver settings) + 
  structural validate() (name uniqueness across ALL categories, 
  ASCII-only identifiers matching equation_parser.py's charset 
  whitelist, action bounds ordering, hard caps on n_sessions/
  max_du_iterations so a hand-edited config can't request an unbounded 
  run). 22 tests.
- equation_parser.py -- safe sympy-based equation parsing. NEVER calls 
  eval()/exec() on raw user text directly, but flagged and closed a REAL 
  sandbox-escape path found empirically while building this: sympy's 
  own `parse_expr` calls Python's real `eval()` internally, and a 
  "safe-looking" restricted local_dict/global_dict does NOT stop 
  attribute access on an already-real object -- confirmed directly that 
  the literal string `"().__class__.__bases__"` parses successfully and 
  returns the actual Python tuple `(object,)` with no dangerous names in 
  scope at all, the classic Python sandbox-escape technique. Closed with 
  two layers verified against a battery of real injection attempts 
  (__import__, os.system, getattr, lambda, chained dunder access, etc. 
  -- all in test_equation_parser.py's parametrized 
  test_injection_attempts_are_rejected_not_executed): (1) a strict 
  ASCII character whitelist rejecting quotes/brackets/braces/semicolons/
  equals/backslash before sympy ever sees the string, (2) rejecting any 
  '.' not sandwiched between two digits (i.e. not part of a plain 
  decimal number) -- since chained attribute access is the ONLY 
  legitimate use of a bare '.' in Python syntax and equations never need 
  it. Also deliberately does NOT use sympy's 
  `implicit_multiplication_application` transform (the "let 2x1 mean 
  2*x1" convenience) despite it looking like an obvious usability win: 
  its bundled `split_symbols` step SILENTLY decomposes an unrecognized 
  multi-letter name into a product of single-character symbols (a typo 
  like "foo" parsed as `f*o**2` instead of raising an error, confirmed 
  by testing) -- equations must write multiplication explicitly. 37 
  tests, including the security battery.
- generic_plant.py -- the ONE generic factory (not per-model generated-
  and-exec'd text) turning a validated config into the four 
  DynamicsFn/MeasurementFn/CostFn/ContinuousDynamicsFn callables, 
  confirmed by diffing dynamic_pricing/ against duopoly_pricing/ first 
  that everything except the per-equation arithmetic and dims is 
  mechanically identical between them. VERIFIED, not just asserted: (1) 
  a wizard config reproducing dynamic_pricing/'s exact math produces 
  numerically identical dynamics/measurement/cost output to the REAL 
  dynamic_pricing module on random inputs; (2) same for duopoly_pricing/ 
  at its harder 2-state/4-unknown/cross-term dims; (3) both compiled 
  plants pass the REAL, unmodified 
  extensions/dual_control/validate_plugin.validate_dual_control_plugin() 
  -- not a wizard-specific stand-in. 16 tests.
- storage.py -- JSON persistence to wizard/models/. Model names are 
  free-form user text turned into filenames -- sanitized against path 
  traversal (`../../etc/passwd`) and Windows-reserved device names 
  (CON/PRN/NUL/etc), verified with dedicated tests, not just assumed 
  safe. 20 tests.
- validation_runner.py / solver_runner.py -- wrap compile_plant() + the 
  real validate_dual_control_plugin()/main_outer_control_loop() with 
  exception translation to plain language at every layer; solver_runner 
  additionally runs the solver in a background thread with a wall-clock 
  timeout (documented limitation: can't forcibly kill a blocking numpy 
  call if it fires, input caps are the real defense) and a post-hoc 
  NaN/Inf sweep over the result. 6 + 10 tests.
- plotting.py / codegen_preview.py -- matplotlib figures + results table 
  from a real solver run; a READ-ONLY human-readable code preview for 
  transparency/"graduating" a model into a real applications/ folder, 
  rendered from the ALREADY-PARSED sympy expression tree (never the 
  user's raw string) and NEVER exec'd by the app itself -- the app only 
  ever runs models via generic_plant.py's closures. 7 + 9 tests.
- app.py -- Streamlit UI orchestration only, 12-step guided flow (dims 
  through cost through advanced settings through review/save/validate/
  run), st.data_editor for every list-shaped step (states/actions/
  parameters/constants/measurement -- avoids hand-rolled dynamic-widget-
  key bugs), live equation feedback on every equation field, st.status 
  spinners around Validate/Run, and a top-level try/except around every 
  step's render call so a bug in one step can't crash the session. 
  MANUALLY VERIFIED end-to-end in a real running browser session (not 
  just unit-tested): loaded duopoly_example, edited a dynamics equation 
  to an unknown name and confirmed the live plain-language error (not a 
  traceback), fixed it, ran Test This Model (pass) and Run Simulation 
  (completed in 1.4s), confirmed all four plot types and the results 
  table rendered with the MODEL'S OWN variable names as column headers, 
  confirmed the parameter-estimate plot faithfully reproduced 
  duopoly_pricing's own documented identifiability finding (own-price 
  sensitivities partially converge, cross-price sensitivities don't -- 
  same real phenomenon, not a wizard bug), confirmed the generated-code 
  preview renders, and separately confirmed on a fresh blank model that 
  the Run Simulation button is genuinely disabled (not just discouraged) 
  until validation passes for the CURRENT config. Zero browser console 
  errors.

**A SECOND real bug found and fixed (beyond the equation-parser security 
finding above), in how the wizard drives the existing, unmodified 
main_outer_control_loop() -- not a bug in that function itself, but a 
real incompatibility the wizard had to route around:** 
main_outer_control_loop's `augment_states_in_filter=False` branch does 
`constants_filter = np.concatenate([constants, p_hat_0])`, which requires 
`constants` to be a raw ndarray -- incompatible with a custom `constants` 
OBJECT (this wizard's GenericConstants, and equally 
PricingConstants/DuopolyPricingConstants) the same way both real demo 
scripts sidestep it by hardcoding augment_states_in_filter=True always. 
But augment_states_in_filter=True has its OWN previously-undiscovered 
edge case (neither real example plant ever exercises n_p=0): it 
unconditionally does `Fp[0, 0]` when re-estimating each session, which 
raises IndexError when Fp is the (0, 0) matrix a zero-unknown-parameter 
model's continuous_dynamics returns. Resolved in solver_runner.py by 
deriving augment_states_in_ilqg/filter directly from `n_p > 0` (never 
independently configurable) and passing `constants` as a bare empty 
ndarray specifically in the n_p=0 case -- safe because n_p=0 means 
generic_plant.py's own dynamics/cost/measurement closures never actually 
read from `constants` regardless of its type. Documented in detail in 
schema.SolverSettings's own docstring since this is exactly the kind of 
non-obvious constraint a future reader would otherwise reintroduce by 
"simplifying" augment_states_in_ilqg/filter back into user-facing 
settings.

Documentation: PYTHON/wizard/WIZARD_GUIDE.md (new, for the no-code 
guided path) written alongside PYTHON/PLUGIN_GUIDE.md (existing, for the 
raw-code path) -- the two cross-reference each other rather than 
duplicating content, per the plan's explicit recommendation and the 
supervisor-facing distinction between "wants guided questions" and 
"comfortable writing Python directly."

Total: 127 wizard-specific tests + 134 pre-existing = 261/261 passing, 
zero regressions.

## Bugs found during conversion (worth telling supervisor)
- Latent symmetry bug in original MATLAB's Sxxh computation 
  (backward_pass.m) — caused inconsistent gradient/Newton-step. Never 
  surfaced in MATLAB (chol() silently reads one triangle) but is a real, 
  pre-existing bug. Fixed in Python port by symmetrizing the matrix.
- (One other formula in Todorov_estimator.m looked asymmetric but was 
  verified correct against the source thesis — not a bug, no fix needed.)

## Katie conversion plan (not started — analysis done 2026-07-28)
Diffed Katie's .m files against Andrew's line-by-line to figure out what's 
reusable before converting. (Target paths below are the post-2026-08-17 
locations in PYTHON/extensions/dual_control/ and PYTHON/core/ddp_solver/ 
— see "Repository layout" above; at the time this plan was written they 
were all still bare filenames destined for PYTHON/andrew/.)

**Reuse as-is (functionally identical to Andrew's, already ported/tested):**
- boxQP.m, makePD.m, pp.m, sabs.m, finite_difference.m — 0 substantive 
  diff -> core/ddp_solver/box_qp.py / extensions/dual_control/make_pd.py / 
  extensions/dual_control/finite_difference.py (inline pp/sabs)
- backward_pass.m, forward_pass.m, Todorov_estimator.m, iLQG_function.m — 
  comment-only diffs -> extensions/dual_control/backward_pass.py / 
  extensions/dual_control/forward_pass.py / 
  extensions/dual_control/todorov_estimator.py / 
  extensions/dual_control/ilqg_function.py
- simulate_system.m — trivial diff (dropped an imaginary-check warning) 
  -> extensions/dual_control/simulate_system.py
- SPKF_function.m — file is 466 lines vs Andrew's 165, but `Mode = 0` is 
  hardcoded at line 12; the extra ~300 lines are a dead `Mode==1` nested-CSP 
  branch that never executes. Call signature matches Andrew's usage 
  exactly -> extensions/dual_control/spkf_function.py already implements 
  the live path.
- None of this solver core is hardcoded to a state/action dimension, so it 
  should work once wired to Katie's 6-state/13-action problem with no new 
  conversion work.

**Needs adaptation (same shape as something ported, but rewritten):**
- Outer_Control_Loop.m vs Main_Outer_Control_Loop.m/
  extensions/dual_control/main_outer_control_loop.py — same 
  session-loop-calling-iLQG-then-SPKF pattern, but ~510 substantively 
  differing lines (persona wiring, swat-ceiling calc, dose bookkeeping, 
  3-vs-13-dim actions). Use main_outer_control_loop.py as a template, not 
  a drop-in. (In the event, this turned out wrong in one respect: 
  applications/sci_rehab/run_katie_pipeline.py reuses 
  main_outer_control_loop.py completely UNMODIFIED as a plant-agnostic 
  MPC wrapper rather than adapting it — see "Repository layout" above.)
- run_sgba_personas.m, run_ilqg_cohort*.m, run_paper_results*.m, 
  compare_two_personas.m, run_persona_comparison.m, run_cohort_sensitivity.m 
  — new top-level orchestration scripts, no direct Andrew analog as 
  standalone scripts, but follow the same iLQG-per-session pattern.

**Genuinely new (no analog to port from):**
- SINDy: build_sindy_library.m, stlsq.m, cross_validate_lambda.m, 
  prune_sindy_bic.m, validate_sindy_model.m, compute_fisher_information.m, 
  print_identified_equations.m. OPEN QUESTION: hand-port STLSQ, or use the 
  existing `pysindy` package (implements this exact method)?
- MATLAB-Stats-Toolbox shims: crossvalind.m, randsample.m, betarnd.m — 
  don't translate literally, use sklearn.model_selection.KFold / 
  numpy.random.choice / numpy.random.default_rng().beta instead.
- Plant/cost: ContinuousStateDynamics.m, DiscreteStateDynamics.m, 
  Measurement.m, l_cost.m — ALL DONE, see "STATUS — Katie conversion" 
  above. Not a regression vs. Andrew's side either, since Andrew's 4 
  equivalent SIDARTHE plant files were also never ported (see below) — 
  no existing Python plant code to compare against regardless. Calling 
  convention matches what simulate_system.py/ilqg_function.py already 
  expect.
- Personas: define_persona.m, who_therapies.m — entirely new.
- generate_synthetic_dataset.m — new (Andrew never generated synthetic 
  cohorts, only ran single trajectories).
- plot_results.m, export_results_json.m — write idiomatically in Python 
  (matplotlib/json), don't translate.
- SKIP, don't port: sgtitle.m/xline.m/yyaxis.m (pure Octave-compat shims — 
  matplotlib already has suptitle/axvline/twinx natively), 
  Synthetic_Data.m/SyntheticDatawNoise.m (unrelated legacy FIM/BBS 
  prototype, not called by any pipeline), 
  ParticleFilter_Matlab_function.m (unused/vestigial, never called live).

**SINDy library decision (2026-07-29): hand-port, do NOT use pysindy.**
Checked pysindy's feature set against build_sindy_library.m, stlsq.m, 
cross_validate_lambda.m, and prune_sindy_bic.m before starting conversion.
- pysindy covers the easy, mechanical parts (PolynomialLibrary/FourierLibrary 
  for the candidate library; STLSQ optimizer implements the same 
  sequential-thresholded-least-squares algorithm; SINDYc-style control-input 
  support for state+action cross terms).
- But it has no equivalent for the parts that are actually load-bearing in 
  Katie's implementation: `protected_idx` (terms immune to thresholding), 
  `min_terms` floor, `max_terms` hard cap, the cross_validate_lambda.m 
  custom CV wrapper (data-adaptive lambda floor, degenerate-fold penalty, 
  complexity-penalized tie-break, inverted 1-SE rule), and prune_sindy_bic.m 
  (BIC-based greedy backward elimination) — none of these exist in pysindy 
  or sklearn. They'd have to be hand-written regardless of whether pysindy 
  is used for the library/optimizer core.
- pysindy's STLSQ also normalises columns by L2-norm, not std (Katie's 
  convention), so even the overlapping part wouldn't reproduce her Xi 
  numerically — breaking the MATLAB-vs-Python verification workflow used 
  successfully on Todorov/Andrew (see bug found via that method, above).
- Conclusion: hand-port build_sindy_library.py / stlsq.py / 
  cross_validate_lambda.py / prune_sindy_bic.py directly from the MATLAB, 
  verified against MATLAB output, same as everything else in this project.

## NOT YET PORTED — decision needed from supervisor
4 SIDARTHE-specific plant files kept as generic callables, not directly 
ported: ContinuousStateDynamics.m, DiscreteStateDynamics.m, 
Measurement.m, l_cost.m. 
QUESTION FOR SUPERVISOR: keep COVID/SIDARTHE dynamics as-is, or replace 
with SCI-relevant dynamics? (The reusable control algorithm — everything 
else — is done; only these 4 "plant" files are COVID-specific.)

**Which thesis figures these 4 files can actually validate against 
(2026-08-25):** cross-checked against the thesis's full figure catalog 
(NOTES/2024_Mathis_Sensinger_Carretero_PhD Thesis.pdf, Ch. 3-5). These 4 
files implement ONE specific variant: ContinuousStateDynamics.m's alpha/
beta/gamma/epsilon/theta formulas are the SECTION 5.2.2 "ACM 
modification" -- 5 named policy controls (u1-u5) combined per-parameter 
via eta effectiveness-factor weights -- not the thesis's original 
single-control SIDARTHE model (the simpler "Kohler 2021 original" 
alpha/gamma formula is present but commented out as dead code in the 
same file). Confirmed exactly by 3 independent numeric matches: 
l_cost.m's cu=1e-2*[1,1000,75,75,50] and cx=10*[0.0033,0.0267], and 
Measurement.m's noise scaling 1e1*diag([1,0.75,0.5,0.25]), are all exact 
matches to Section 5.3.2's stated cu/cx/G for the SIXTEEN-uncertain-
parameter comparison (Figs 5.10-5.13, dual vs adaptive iLQG only -- no 
NMPC baseline there either). Main_Outer_Control_Loop.m runs this exact 
setup with p_true/p_hat matching Table 5.3. 
CONSEQUENCE: Figs 5.10-5.13 are the ONLY figures in the entire thesis 
with a runnable MATLAB reference in this repo -- porting these 4 files 
would let a Python run be diffed against real .m output, same 
verification method used for every other conversion in this project. 
Every other results figure in the thesis (Ch. 3's linear/Rohr's-example 
toy systems, Ch. 4's AM2 digestion model, Ch. 5 Section 5.3.1's original 
single-control 2-parameter SIDARTHE comparison, and every MS-SP-NMPC/
MRAC baseline series throughout) has NO corresponding .m file anywhere 
in CODES/ -- those baselines are external published methods to begin 
with, not Andrew's own implementation, so reproducing them would mean 
coding from the thesis's written equations alone with nothing to diff 
against. 
DISCREPANCY FLAGGED (not reconciled): Main_Outer_Control_Loop.m sets 
T=30, dt=1 -> N=30 steps, but the thesis's own Section 5.3.2 body text 
states 40 steps x 1 day for this same experiment. Worth checking which 
one the actual published Figs 5.10-5.13 were rendered from before 
treating either as ground truth for a Python diff. 
Full figure-by-figure catalog (axis labels, legend text, equations, 
parameter values) published as an artifact -- ask to have the link 
resurfaced if needed, or regenerate from the thesis PDF directly.

## Still needed
- Zahid's code — no original MATLAB/code was ever found; see "STATUS — 
  Zahid conversion" above — the from-paper Python implementation is now 
  feature-complete (both variants ported, wired into the real solver, 
  zero regressions, verified end-to-end against the paper's own worked 
  example)
- Decision on SIDARTHE vs SCI dynamics (see above) — Katie's code 
  answers this in practice: it already replaced SIDARTHE with SCI 
  dynamics, and the full Python SCI plant (continuous_state_dynamics.py/
  discrete_state_dynamics.py/measurement.py/l_cost.py) is now ported 
  too, so this just needs supervisor sign-off rather than a from-scratch 
  decision
- Katie's code: see "STATUS — Katie conversion", "STATUS — Katie SINDy 
  pipeline", "STATUS — Katie personas", and "STATUS — 
  generate_synthetic_dataset.py" above for what's done so far -- 
  reuse-as-is solver core confirmed at Katie's dims, SCI plant/cost math 
  fully ported, the full 7-file SINDy pipeline complete, personas done, 
  AND the synthetic-data generator done. CORRECTION (found during the 
  2026-08-17 reorganization, not previously recorded here): the 
  top-level orchestration script this bullet used to call "still 
  needed" already existed as of 2026-07-30 -- 
  applications/sci_rehab/run_katie_pipeline.py wires the full chain 
  (generate data -> build_sindy_library -> cross_validate_lambda -> 
  stlsq -> prune_sindy_bic -> validate_sindy_model -> 
  compute_fisher_information -> print_identified_equations -> dual-iLQG 
  control via extensions/dual_control/main_outer_control_loop.py, 
  reused unmodified) end to end, per its own docstring's "STRUCTURE, 
  adapted from Main_Outer_Control_Loop.m" section. 
  applications/sci_rehab/run_katie_pipeline_preview.py is its cheap 
  dry-run counterpart. Both import-smoke-tested clean post-move; 
  run_katie_pipeline.py itself has no dedicated automated test (it's a 
  script, not a test module) but every function it composes is 
  independently tested. This was simply never reflected back into this 
  "Still needed" section when it was written -- Python 
  implementation-wise, Katie's conversion is essentially complete.

## Conventions
- Python version: 3.14 (Windows, C:\Users\fatim\AppData\Local\Python\
  pythoncore-3.14-64) — confirmed 2026-08-17.
- Key libraries used: numpy, scipy, matplotlib, scikit-learn, pytest 
  (used for cross_validate_lambda.py's KFold — replaces MATLAB Stats 
  Toolbox's crossvalind, per the Katie conversion plan's decision). 
  Originally installed 2026-07-29; wiped by a Python version update at 
  some point before 2026-08-17 and reinstalled that day during the 
  repository reorganization below.
- Package layout (since 2026-08-17): PYTHON/ is organized as 
  core/extensions/applications (see "Repository layout" above), with 
  every directory a real package (__init__.py) and cross-module imports 
  written as absolute package imports, e.g. 
  `from core.ddp_solver.ilqg import ilqg`. PYTHON/pyproject.toml + 
  `pip install -e .` (run once from PYTHON/) registers the package so 
  this resolves regardless of cwd — re-run `pip install -e .` from 
  PYTHON/ if imports of core/extensions/applications start failing in a 
  fresh environment. Tests run per-package 
  (`python -m pytest PYTHON/<package>/<subpackage>`) or as the whole 
  tree in one invocation (`python -m pytest PYTHON/`) — both work now 
  that the old same-named-module collision is gone.
- MATLAB is 1-indexed, Python is 0-indexed — checked carefully during conversion
- MATLAB `*` = matrix multiply; NumPy `*` = element-wise, used `@` instead

## What I want help with
- Explaining code/math clearly, not just restating it
- Verifying every conversion with tests before moving to the next piece
- Flagging discrepancies between MATLAB and Python rather than silently fixing