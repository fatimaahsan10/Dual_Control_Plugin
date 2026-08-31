/* =====================================================================
   STUDY_GUIDE/content.js
   All factual content for the study dashboard. Every number, filename,
   and finding here is taken directly from CLAUDE.md and from the
   migration work performed in this project (test runs actually executed,
   files actually moved). Where something is uncertain or not directly
   verified, it is labeled "UNCERTAIN" or "not verified" rather than
   stated as fact.
   ===================================================================== */

const CHAPTERS = [
{
  id: "overview", nav: "Overview", icon: "01", deep: false,
  title: "Project Overview",
  html: `
    <p class="lede">Mitacs internship, University of New Brunswick. Goal: apply
    <b>curiosity-driven AI</b> — intrinsic motivation / curiosity-driven reinforcement
    learning — to spinal cord injury (SCI) rehab and motor recovery modeling.</p>

    <div class="callout beginner">
      <div class="callout-label">Simple version</div>
      <p>Imagine a physiotherapist deciding, week by week, which of 13 possible therapies
      to give a patient. A normal AI would need someone to hand it a perfect model of how
      the patient's body responds to therapy before it can plan anything. A
      <b>curiosity-driven</b> AI is different: it is also allowed to try things partly
      because they'll teach it something about the patient it doesn't know yet — it
      controls <i>and</i> learns at the same time.</p>
    </div>

    <div class="callout technical">
      <div class="callout-label">Technical version</div>
      <p>This project builds toward a <b>dual-control</b> framework: a controller that
      jointly (1) identifies unknown system dynamics from data and (2) selects actions
      that balance minimizing cost against reducing uncertainty about the model itself.
      The concrete instrument for "identify dynamics from data" is
      <b>SINDy</b> (Sparse Identification of Nonlinear Dynamics); the concrete instrument
      for "plan under uncertainty" is <b>DDP/iLQG</b> extended with parameter estimation
      (SPKF) and, separately, hard safety constraints.</p>
    </div>

    <h3>Four codebases, one destination</h3>
    <p>None of these four pieces of work were built together. They were four separate
    contributions — three MATLAB codebases received from different people, plus one paper
    with no code at all — and the task was to understand each one, port or build it in
    Python with tests proving correctness, and then organize all four into one coherent
    framework.</p>

    <div class="grid-4">
      <div class="mini-card person-todorov">
        <div class="mini-card-name">Todorov</div>
        <div class="mini-card-role">The foundation</div>
        <div class="mini-card-desc">A generic, deterministic optimal-control solver
        (DDP/iLQG). Not SCI-specific at all — just the math engine.</div>
      </div>
      <div class="mini-card person-andrew">
        <div class="mini-card-name">Andrew</div>
        <div class="mini-card-role">Makes it handle uncertainty</div>
        <div class="mini-card-desc">Extends Todorov's solver to work when the system is
        noisy and its parameters are partly unknown — dual control.</div>
      </div>
      <div class="mini-card person-katie">
        <div class="mini-card-name">Katie</div>
        <div class="mini-card-role">Makes it about SCI rehab</div>
        <div class="mini-card-desc">Replaces a COVID epidemic model with a real SCI
        patient model, adds SINDy to learn dynamics from data, adds 13 WHO therapies and
        18 patient personas.</div>
      </div>
      <div class="mini-card person-zahid">
        <div class="mini-card-name">Zahid</div>
        <div class="mini-card-role">Makes it safe</div>
        <div class="mini-card-desc">A paper-only method (no code existed) for making the
        solver respect hard constraints, like "this angle must never exceed 1."</div>
      </div>
    </div>

    <h3>How they connect, in one sentence each</h3>
    <ol class="connect-chain">
      <li><b>Todorov</b> provides the solver that decides a sequence of actions.</li>
      <li><b>Andrew</b> takes that solver and adds the ability to plan under uncertainty
      and learn system parameters while acting.</li>
      <li><b>Katie</b> reuses Andrew's uncertainty-aware solver <i>unmodified</i>, but
      replaces the plant (what's being controlled) with a real SCI patient model, and adds
      SINDy so the "known" dynamics are actually learned from data instead of assumed.</li>
      <li><b>Zahid</b> is independent of the other three's chain — he extends
      <i>Todorov's</i> original solver directly, adding the ability to enforce hard safety
      constraints, and is not (yet) wired into Katie's SCI pipeline.</li>
    </ol>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>One goal: curiosity-driven AI (control + learn together) applied to SCI rehab.</li>
        <li>Four separate contributions, ported/built in Python, now organized as one
        framework.</li>
        <li>The chain is Todorov &rarr; Andrew &rarr; Katie for the main SCI pipeline; Zahid
        is a separate, independent extension of Todorov's solver.</li>
      </ol>
      <div class="one-liner">"This project takes four independent pieces of control-theory
      code — a generic solver, an uncertainty-aware extension, an SCI-specific rehab
      model with data-driven identification, and a hard-constraint extension — and turns
      them into one tested, organized Python framework."</div>
    </div>
  `
},

{
  id: "timeline", nav: "Timeline", icon: "02", deep: false,
  title: "What I Did — Timeline",
  html: `
    <p class="lede">Roughly in this order (dates as recorded in the project log,
    CLAUDE.md):</p>
    <div class="timeline">
      <div class="tl-row"><div class="tl-date">Received</div>
        <div class="tl-text">Todorov's MATLAB toolbox, Andrew's MATLAB dual-control code,
        Katie's MATLAB codebase (~46 files, zip, 2026-07-28), and Zahid's paper (no code).</div></div>
      <div class="tl-row"><div class="tl-date">Todorov</div>
        <div class="tl-text">Ported the 4-file deterministic DDP/iLQG solver to Python.
        Verified exact match against an independently-derived LQR solution. <b>6/6 tests.</b></div></div>
      <div class="tl-row"><div class="tl-date">Andrew</div>
        <div class="tl-text">Ported the 7-file dual-control extension (stochastic dynamics,
        SPKF parameter estimation, MPC loop). Verified a deliberately-wrong parameter guess
        (0.4) converges to the true value (1.0) while cost drops.</div></div>
      <div class="tl-row"><div class="tl-date">2026-07-28 &rarr; 2026-07-30</div>
        <div class="tl-text">Diffed Katie's MATLAB files against Andrew's line-by-line to
        plan the conversion. Confirmed Andrew's solver core is reusable <i>as-is</i> at
        Katie's dimensions (6 states, 13 actions) — just needed new tests at the new size,
        not new code. Ported Katie's 4 SCI plant/cost files, then the 7-file SINDy pipeline,
        then personas (who_therapies.py, define_persona.py), then the synthetic-data
        generator.</div></div>
      <div class="tl-row"><div class="tl-date">2026-07-30</div>
        <div class="tl-text">Built <code>run_katie_pipeline.py</code> — the first script
        that runs the <i>entire</i> chain end to end: generate data &rarr; identify a SINDy
        model &rarr; validate it &rarr; run dual-control therapy planning with the
        identified model.</div></div>
      <div class="tl-row"><div class="tl-date">2026-08-04</div>
        <div class="tl-text">Built Zahid's constraint method from the paper's equations
        (no source code existed). Wired it into Todorov's solver as a small, optional
        extension. Found and root-caused a real convergence trap at the paper's own
        alpha=0.1, then confirmed alpha=1.0 qualitatively reproduces the paper's figure.</div></div>
      <div class="tl-row"><div class="tl-date">2026-08-17</div>
        <div class="tl-text"><b>Reorganized</b> all four codebases from four flat,
        per-person folders into one shared <code>core / extensions / applications</code>
        framework — moved every file, rewrote every import to a real Python package import,
        re-ran the full test suite after every group of moves. Result: <b>132/132 tests
        passing</b>, and — for the first time — the whole tree runs as a single
        <code>pytest</code> invocation (previously impossible due to a module-name
        collision). Updated CLAUDE.md to match.</div></div>
      <div class="tl-row"><div class="tl-date">Today</div>
        <div class="tl-text">Built this study dashboard for the presentation.</div></div>
    </div>
    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Order: Todorov &rarr; Andrew &rarr; Katie (plant + SINDy + personas) &rarr;
        Zahid &rarr; full reorganization.</li>
        <li>Every conversion was verified with tests before moving to the next piece.</li>
        <li>The reorganization is the most recent, largest single change: four folders
        &rarr; three layers, zero regressions.</li>
      </ol>
      <div class="one-liner">"I converted and tested each codebase in order, then spent the
      final phase reorganizing everything into one layered framework with zero test
      regressions."</div>
    </div>
  `
},

{
  id: "todorov", nav: "Todorov", icon: "03", deep: true,
  title: "Todorov — The Foundation Solver",
  html: `
    <div class="callout beginner">
      <div class="callout-label">Simple version</div>
      <p>Imagine you're driving a car into a tight parking spot. You don't know the
      perfect sequence of steering/gas moves in advance — you plan a rough attempt, see
      how close it gets you, then adjust, over and over, until you're parked well. Todorov's
      code is a generic version of that adjust-and-improve process, for any system you can
      describe mathematically.</p>
    </div>

    <div class="callout technical">
      <div class="callout-label">Technical version</div>
      <p><b>DDP</b> (Differential Dynamic Programming) and its simplified cousin
      <b>iLQG</b> (iterative Linear-Quadratic-Gaussian) are trajectory-optimization
      algorithms. Given a starting state and a rough guess at a sequence of controls,
      they repeatedly: (1) locally approximate the dynamics and cost as quadratic around
      the current trajectory, (2) solve that local approximation backward in time
      (the <b>backward pass</b>) to get a locally-optimal correction, (3) apply the
      correction forward in time (the <b>forward pass</b>) with a line search, and
      (4) repeat until the trajectory stops improving. "Control-limited" means the solver
      respects hard bounds on the controls (e.g. a motor can't exceed max torque) via a
      constrained QP solve inside the backward pass.</p>
    </div>

    <h3>Four key words, with an example</h3>
    <div class="jargon-table">
      <div class="jargon-row"><div class="jargon-term">State</div>
        <div class="jargon-def">Where the system currently is. <i>Example (pendulum):</i>
        current angle and angular velocity.</div></div>
      <div class="jargon-row"><div class="jargon-term">Control</div>
        <div class="jargon-def">The action you apply. <i>Example:</i> the torque you push
        the pendulum with right now.</div></div>
      <div class="jargon-row"><div class="jargon-term">Dynamics</div>
        <div class="jargon-def">The rule for how the state changes given the current state
        and control. <i>Example:</i> the pendulum's physics equations.</div></div>
      <div class="jargon-row"><div class="jargon-term">Cost</div>
        <div class="jargon-def">A number saying how "bad" a state/control is right now
        (lower is better). <i>Example:</i> distance from the goal, plus a penalty for using
        too much force.</div></div>
    </div>

    <h3>The four files</h3>
    <table class="module-table">
      <thead><tr><th>File</th><th>Job</th></tr></thead>
      <tbody>
        <tr><td class="mono">box_qp.py</td><td>Solves the "which correction respects my
        control limits?" sub-problem used inside the backward pass.</td></tr>
        <tr><td class="mono">back_pass.py</td><td>The backward pass — computes the local
        quadratic correction to the trajectory, timestep by timestep, backward from the
        end.</td></tr>
        <tr><td class="mono">forward_pass.py</td><td>The forward pass — applies the
        correction, with a line search over step sizes, to actually get a new (better)
        trajectory.</td></tr>
        <tr><td class="mono">ilqg.py</td><td>The outer loop: repeatedly calls
        back_pass then forward_pass until convergence. This is the function everything
        else calls.</td></tr>
      </tbody>
    </table>

    <h3>What was converted, and how it was tested</h3>
    <p>All four files were ported from MATLAB to Python
    (<code>core/ddp_solver/</code> today). Correctness was checked the strongest way
    available: for a linear system with quadratic cost, iLQG's local quadratic
    approximation is <i>exact</i>, so a correct implementation must converge to the exact
    same answer as directly solving the textbook Riccati/LQR equations — an
    independently-derived ground truth, not just "the numbers look reasonable."
    <b>Result: exact match, 6/6 tests passing.</b></p>

    <div class="callout warn">
      <div class="callout-label">⚠️ Important caveat / real finding</div>
      <p>While converting <code>back_pass.py</code>, a real bug was found in the
      <i>original MATLAB</i>: a matrix (<code>Sxxh</code>) that mathematically must be
      symmetric was computed asymmetrically. MATLAB's Cholesky solver (<code>chol()</code>)
      silently only reads one triangle of the matrix, so the bug never surfaced there — but
      it's a genuine pre-existing bug, not a Python-conversion artifact. Fixed in the Python
      port by explicitly symmetrizing the matrix.</p>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Todorov = the generic optimizer: state, control, dynamics, cost &rarr;
        best action sequence, via repeated backward/forward passes.</li>
        <li>Verified against an exact closed-form solution (LQR), not just "runs fine."</li>
        <li>Found a real latent symmetry bug in the original MATLAB, invisible there only
        because of how MATLAB's solver happens to read matrices.</li>
      </ol>
      <div class="one-liner">"Todorov's code is the generic optimal-control engine
      underneath everything else — I ported it and proved it exactly matches a
      closed-form LQR solution, and found a real symmetry bug in the original MATLAB
      along the way."</div>
    </div>
  `
},

{
  id: "andrew", nav: "Andrew", icon: "04", deep: true,
  title: "Andrew — Controlling While Learning",
  html: `
    <div class="callout beginner">
      <div class="callout-label">Simple version</div>
      <p>Todorov's planner assumes you already know exactly how the system behaves. Real
      systems are noisy, and you often don't know some of their parameters precisely.
      <b>Dual control</b> simply means: control the system while learning about the system
      at the same time, instead of learning first and controlling second.</p>
    </div>

    <div class="callout technical">
      <div class="callout-label">Technical version</div>
      <p>Andrew's extension adds three things on top of Todorov's deterministic core:
      (1) <b>stochastic dynamics</b> — the system now has random noise; (2)
      <b>parameter estimation</b> via <b>SPKF</b> (a sigma-point/unscented-style Kalman
      filter) that maintains a belief (mean + covariance) over unknown parameters and
      updates it from each new observation; (3) <b>MPC</b> (receding-horizon control) — at
      every step, re-plan a full trajectory with the current best estimate, take only the
      first action, observe the result, re-estimate, and repeat.</p>
    </div>

    <h3>Tiny example</h3>
    <p>Suppose the true "strength" of a motor is 1.0, but you start out believing it's
    0.4. A dual controller doesn't just act on the wrong belief forever — because its
    actions also generate data, and the SPKF estimator uses that data to correct the
    belief toward 1.0 over time, while the controller's plan updates to match. That is
    exactly what was verified in testing (see below).</p>

    <h3>The 7 files</h3>
    <table class="module-table">
      <thead><tr><th>File</th><th>Job</th></tr></thead>
      <tbody>
        <tr><td class="mono">backward_pass.py / forward_pass.py</td><td>Same role as
        Todorov's, but reimplemented to handle the stochastic/augmented-state case
        (not a copy — an independent reimplementation of the same algorithm).</td></tr>
        <tr><td class="mono">todorov_estimator.py</td><td>An estimator component
        used alongside the solver.</td></tr>
        <tr><td class="mono">ilqg_function.py</td><td>The outer solve loop for this
        stochastic/estimation-aware version (equivalent role to Todorov's ilqg.py).</td></tr>
        <tr><td class="mono">make_pd.py</td><td>Utility: forces a matrix to be positive
        definite (needed for covariance matrices to stay valid).</td></tr>
        <tr><td class="mono">spkf_function.py</td><td>The sigma-point Kalman filter —
        the actual "learn the parameters" machinery.</td></tr>
        <tr><td class="mono">main_outer_control_loop.py</td><td>The session-by-session
        MPC loop: plan &rarr; step the real system &rarr; re-estimate &rarr; warm-start
        &rarr; repeat. Written generically (dynamics/measurement/cost passed in as
        arguments) — this turns out to matter a lot later (see Katie chapter).</td></tr>
      </tbody>
    </table>

    <h3>How it was tested</h3>
    <p>The strongest test here isn't a closed-form solution (dual control doesn't have
    a simple one) — it's a <b>behavioral</b> test: start the estimator deliberately wrong
    (0.4 instead of the true 1.0) and confirm the estimate genuinely converges toward the
    true value while cost drops over iterations. That confirms the dual-control mechanism
    is doing real work, not just passing inputs through unchanged.</p>

    <div class="callout warn">
      <div class="callout-label">⚠️ Number to double-check</div>
      <p>CLAUDE.md's written status for Andrew says "24/24 tests passing." Running the
      actual current test suite for this code (now <code>extensions/dual_control/</code>)
      gives <b>16/16 passing</b>, confirmed multiple times during the reorganization. This
      is a real, unresolved discrepancy between the documentation and the live suite —
      flagged here rather than picking one number silently. If asked for an exact number,
      16 is the one directly verified by actually running the tests today.</p>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Dual control = plan and learn parameters simultaneously, not sequentially.</li>
        <li>SPKF maintains a belief over unknown parameters; MPC re-plans every step using
        the current belief.</li>
        <li>Tested behaviorally: a deliberately wrong parameter guess converges to the
        true value while cost drops.</li>
      </ol>
      <div class="one-liner">"Andrew's code extends Todorov's solver so it can plan under
      uncertainty and learn unknown system parameters online — I verified this by starting
      an estimate deliberately wrong and confirming it converges to the truth while
      controlling."</div>
    </div>
  `
},

{
  id: "katie", nav: "Katie", icon: "05", deep: true,
  title: "Katie — The Actual SCI Rehab Model",
  html: `
    <div class="callout beginner">
      <div class="callout-label">Simple version</div>
      <p>Andrew's code was tested on a COVID epidemic model — useful as a testbed, but
      not about rehab at all. Katie's contribution swaps that out for a real spinal cord
      injury patient: a small set of numbers describing the patient's condition, a list of
      13 real physiotherapy interventions the World Health Organization defines, and 18
      different patient "personas" who respond differently to those interventions.</p>
    </div>

    <div class="callout technical">
      <div class="callout-label">Technical version</div>
      <p>Katie's codebase reuses Andrew's dual-control solver core <i>completely
      unmodified</i> (confirmed by diffing every file line-by-line against Andrew's), and
      replaces only the "plant" — the state space, dynamics, and cost function — with a
      genuine SCI model. She also adds <b>SINDy</b> (its own chapter) so the dynamics
      aren't assumed known, but identified from simulated patient data.</p>
    </div>

    <h3>The SCI state, with an example</h3>
    <p>Instead of epidemic variables (infected, recovered, ...), the state now has 6
    entries: <b>SCIM</b> and <b>BBS</b> (two functional-recovery scores that actually
    change over time), plus <b>AIS</b> (injury severity grade), <b>Age</b>, <b>DPI</b>,
    and <b>Caregiver support</b> (mostly static per-patient facts). The "control" is a
    13-length vector: how much of each of 13 WHO therapies (like balance training, gait
    training, strengthening) to prescribe this week.</p>

    <div class="flow-mini">
      <div class="flow-step">Patient state (6 numbers)</div><div class="flow-arrow">&darr;</div>
      <div class="flow-step">Therapist chooses doses for 13 therapies</div><div class="flow-arrow">&darr;</div>
      <div class="flow-step">Patient's SCIM/BBS scores change a bit, based on the dose and
      <i>this patient's own</i> responsiveness</div>
    </div>

    <h3>The 4 "plant/cost" files ported</h3>
    <table class="module-table">
      <thead><tr><th>File</th><th>Job</th></tr></thead>
      <tbody>
        <tr><td class="mono">continuous_state_dynamics.py</td><td>The instantaneous rate
        of change of the state, given the current state/actions and the identified/assumed
        model coefficients.</td></tr>
        <tr><td class="mono">discrete_state_dynamics.py</td><td>Turns that instantaneous
        rate into a one-week step (Euler integration) plus process noise — this is the
        actual function the solver calls at each timestep.</td></tr>
        <tr><td class="mono">measurement.py</td><td>Models what's actually observed
        (vs. the true internal state) — e.g. measurement noise.</td></tr>
        <tr><td class="mono">l_cost.py</td><td>The cost function: rewards functional
        recovery, penalizes therapy burden/resources, per-persona weighted.</td></tr>
      </tbody>
    </table>

    <h3>Personas — why they matter</h3>
    <p>18 personas total: 5 population archetypes (e.g. <i>frail_elder</i>,
    <i>young_athlete</i>) + 12 named, concrete SGBA+ (Sex- and Gender-Based Analysis Plus)
    patients + 1 population-average default. Each persona has its own
    <code>therapy_efficacy</code> — how well <i>this</i> patient responds to each of the 13
    therapies. Crucially, this is baked into the <i>simulated physiology itself</i>
    (<code>generate_synthetic_dataset.py</code>), not into the cost function — so the
    controller has to genuinely discover a persona's best therapy mix from how their body
    actually responds, it isn't just told the answer.</p>

    <div class="callout beginner">
      <div class="callout-label">Concrete result</div>
      <p>Running the demo on two personas back to back: the <b>default</b> persona
      front-loads dose early and favors balance/gait/fitness training, reaching its
      recovery ceiling by session 5. The <b>frail_elder</b> persona — far more
      burden-averse, and a poor responder to fitness training specifically — doses much
      more conservatively and picks a different third-favorite therapy
      (functional positioning instead of fitness). Same controller, same code — genuinely
      different behavior because the persona's actual physiology differs.</p>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Katie reuses Andrew's solver unmodified; she only replaces the plant
        (COVID &rarr; SCI) and cost.</li>
        <li>6-number SCI state, 13-therapy WHO action space, 18 personas.</li>
        <li>Persona differences live in the simulated physiology, so the controller
        genuinely has to discover them, not follow hand-coded rules.</li>
      </ol>
      <div class="one-liner">"Katie's contribution is the actual clinical model — a real
      6-variable SCI patient state, 13 WHO therapies, and 18 personas whose different
      physiological responses genuinely change what the controller recommends."</div>
    </div>
  `
},

{
  id: "sindy", nav: "SINDy", icon: "06", deep: true,
  title: "SINDy — Learning the Equations From Data",
  html: `
    <div class="callout beginner">
      <div class="callout-label">Simple version</div>
      <p>Normally, to plan actions you need someone to hand you the equations describing
      how the system behaves. <b>SINDy</b> (Sparse Identification of Nonlinear Dynamics)
      instead <i>discovers</i> those equations by watching data — it looks at many
      patients' weekly states and the therapies they got, and figures out which small set
      of mathematical terms best explains how SCIM and BBS actually changed.</p>
    </div>

    <div class="callout technical">
      <div class="callout-label">Technical version</div>
      <p>SINDy builds a large library of <i>candidate</i> terms (polynomials and
      cross-terms of states and actions), then finds a <b>sparse</b> linear combination of
      those terms — most coefficients exactly zero — that best predicts the observed
      one-step change in state. "Sparse" matters clinically: a dense model with all 200+
      terms active is unreadable and almost certainly overfit; a sparse model might say
      "SCIM mostly depends on 3-4 real drivers," which a clinician can actually read and
      trust.</p>
    </div>

    <h3>The pipeline, in order</h3>
    <div class="flow-mini">
      <div class="flow-step"><b>build_sindy_library.py</b><br><span class="flow-sub">builds
      the big candidate-terms table (Theta) and the target derivatives (dX)</span></div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-step"><b>cross_validate_lambda.py</b><br><span class="flow-sub">picks
      the sparsity threshold via 5-fold cross-validation</span></div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-step"><b>stlsq.py</b><br><span class="flow-sub">Sequential
      Thresholded Least Squares — the actual sparse solve, at the chosen threshold</span></div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-step"><b>prune_sindy_bic.py</b><br><span class="flow-sub">second
      pass: drops any surviving term whose complexity cost (BIC) isn't earning its
      keep</span></div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-step"><b>validate_sindy_model.py</b><br><span class="flow-sub">simulates
      the identified model forward on held-out patients; accepts if error (NMAE) &lt;
      1.0</span></div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-step"><b>compute_fisher_information.py</b><br><span class="flow-sub">how
      confident can we be in each coefficient? Feeds the dual controller's
      "curiosity"</span></div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-step"><b>print_identified_equations.py</b><br><span class="flow-sub">prints
      the final sparse equations in clinician-readable form (SCIM, BBS, named
      therapies)</span></div>
    </div>

    <h3>Why hand-port this instead of using the <code>pysindy</code> Python package?</h3>
    <p>A real package for exactly this method already exists. It was checked and rejected
    for three concrete reasons:</p>
    <ul class="reason-list">
      <li><b>Missing features that are load-bearing here</b>: pysindy has no equivalent
      of <code>protected_idx</code> (terms immune to thresholding — e.g. always keep the
      action terms), <code>min_terms</code>/<code>max_terms</code> (floor/cap on model
      size), the custom cross-validation wrapper, or BIC-based pruning. These would all
      have to be hand-written regardless.</li>
      <li><b>Different normalization</b>: pysindy normalizes columns by L2-norm; Katie's
      MATLAB convention normalizes by standard deviation. Using pysindy would not
      reproduce the MATLAB numbers, breaking the verification approach used successfully
      elsewhere in this project.</li>
      <li>Conclusion: hand-port, verify directly against the MATLAB source's own output —
      same standard as every other conversion in this project.</li>
    </ul>

    <div class="callout warn">
      <div class="callout-label">⚠️ Real bugs this hand-port caught</div>
      <p><b>ddof=1 trap:</b> MATLAB's <code>std()</code> defaults to dividing by N-1
      (sample standard deviation); NumPy's <code>.std()</code> defaults to dividing by N
      (population). Getting this wrong would have silently rescaled every column's
      normalization — and, downstream, every thresholding decision — by a factor of
      &radic;(N/(N-1)). Caught and fixed with an explicit <code>ddof=1</code> in five
      separate files across this pipeline.</p>
      <p><b>NaN-handling trap:</b> MATLAB's <code>max(x, 0.005)</code> ignores NaN;
      NumPy's plain <code>max()</code>/<code>np.maximum</code> propagate it, which would
      have silently corrupted a grid-trimming comparison. Fixed with <code>np.fmax</code>
      (NaN-ignoring, matching MATLAB) instead.</p>
    </div>

    <h3>Tested how</h3>
    <p><b>31 tests</b> across the 7 files, including end-to-end composition checks running
    the real pipeline at Katie's actual dimensions (6 states, 13 actions) — e.g. one test
    deliberately runs at a dimension where the candidate library (210 terms) is
    <i>larger</i> than the number of training rows (200), specifically to exercise the
    numerically-unstable case compute_fisher_information.py has to regularize for.</p>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>SINDy = discover sparse governing equations from data, instead of assuming
        them.</li>
        <li>Pipeline: build library &rarr; CV lambda &rarr; sparse solve (STLSQ) &rarr;
        BIC-prune &rarr; validate &rarr; Fisher info &rarr; print equations.</li>
        <li>Hand-ported (not pysindy) because protected/min/max-term logic and BIC pruning
        don't exist in the package, and its normalization convention differs.</li>
      </ol>
      <div class="one-liner">"SINDy replaces 'assume you know the dynamics' with 'discover
      a sparse, clinician-readable set of governing equations from patient data' — I
      hand-ported the whole 7-stage pipeline because the package version is missing the
      exact custom logic this project depends on."</div>
    </div>
  `
},

{
  id: "zahid", nav: "Zahid", icon: "07", deep: true,
  title: "Zahid — Enforcing Hard Constraints",
  html: `
    <div class="callout beginner">
      <div class="callout-label">Simple version</div>
      <p>Some systems have lines that must never be crossed — a pendulum that must not
      swing past a certain angle, a robot arm that must not hit a wall. Zahid's method
      teaches the DDP solver to respect limits like that, by tightening the allowed
      control range at every single timestep based on how close the current trajectory is
      to the danger zone.</p>
    </div>

    <div class="callout technical">
      <div class="callout-label">Technical version</div>
      <p>Extends control-limited DDP to general inequality constraints
      h(x,u) &ge; 0 or h(x) &ge; 0, <b>without touching the Q/V-function recursion at
      all</b> — the only change is that the box bounds fed into the existing box-QP solver
      are recomputed every outer iteration from the current trajectory, instead of being
      one fixed constant. Two variants: <b>Variant A</b> handles constraints that already
      involve the control directly; <b>Variant B</b> handles pure state constraints by
      first reducing them to a control constraint via one Lie derivative plus a
      control-barrier-function-style decay term (&alpha;&middot;h(x)), then reusing
      Variant A's machinery.</p>
    </div>

    <div class="callout warn">
      <div class="callout-label">⚠️ Different situation from the other three</div>
      <p>Zahid's actual codebase was never received, and no public repository exists for
      it. This is the <b>only</b> one of the four that is not a MATLAB-to-Python
      <i>port</i> — it's an original implementation built directly from the equations in
      his paper (Dastan &amp; Sensinger, 2024, IEEE 63rd CDC). Verification is against the
      paper's own worked example (hand-derived closed forms, qualitative match to its
      reported figure), <b>not</b> bit-exact numeric parity like the other three — a real,
      disclosed limitation of this branch.</p>
    </div>

    <h3>The 2 files</h3>
    <table class="module-table">
      <thead><tr><th>File</th><th>Job</th></tr></thead>
      <tbody>
        <tr><td class="mono">dynamic_control_bounds.py</td><td>Variant A: given a
        constraint and the current trajectory point, solves for the tightened control
        bound at that point (first-order/Jacobian-based).</td></tr>
        <tr><td class="mono">relative_degree_reduction.py</td><td>Variant B: turns a
        pure state constraint into a control constraint (Lie derivative + decay term), then
        hands it to Variant A.</td></tr>
      </tbody>
    </table>
    <p>Both were then wired into Todorov's solver itself — <code>core/ddp_solver/</code>'s
    <code>back_pass.py</code>, <code>forward_pass.py</code>, and <code>ilqg.py</code> each
    got a small, optional, backward-compatible extension (a per-timestep bound array and
    an optional <code>constraint_fn</code> hook) rather than a wrapper layer. Existing
    call sites that don't use constraints are completely untouched.</p>

    <h3>⭐ The headline finding — worth knowing cold</h3>
    <div class="finding-card">
      <div class="finding-part"><span class="finding-label">Expected</span>
      Running the paper's own inverted-pendulum example with the paper's own stated
      &alpha;=0.1 should converge in about 16 iterations to a cost near 10.11 (the paper's
      reported result).</div>
      <div class="finding-part"><span class="finding-label">Observed</span>
      With the literal &alpha;=0.1, the solver gets stuck after one accepted step — cost
      stuck at 149.6, exits by hitting its regularization ceiling (11 iterations, not
      converging).</div>
      <div class="finding-part"><span class="finding-label">Meaning</span>
      Root-caused, not a bug: at small &alpha;, the allowed control range is genuinely
      razor-thin for the first ~0.14 radians of the swing, leaving the solver almost no
      room to move — confirmed by direct inspection, and confirmed to be a real property
      of the method (an &alpha; sweep at 0.3 still traps <i>and</i> the trajectory actually
      violates the constraint; &alpha;=1.0 converges cleanly in 13 iterations to cost
      11.46, staying under the limit — a genuine, honest qualitative match to the paper's
      figure). This directly demonstrates a limitation the paper's own text admits
      ("as long as the adjustments... are modest").</div>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Zahid's method tightens control bounds every iteration based on the current
        trajectory — no change to the core solver math.</li>
        <li>No original code existed — built directly from the paper's equations, verified
        against its own hand-derived example.</li>
        <li>The paper's literal &alpha;=0.1 traps the solver — a real, root-caused finding,
        not a bug; &alpha;=1.0 reproduces the paper's result honestly.</li>
      </ol>
      <div class="one-liner">"Zahid's paper had no code at all, so I implemented the
      constrained-DDP method directly from its equations and verified it against the
      paper's own worked example — and found that its literal alpha value actually traps
      the solver, a real and root-caused finding, not a bug."</div>
    </div>
  `
},

{
  id: "connections", nav: "Connections", icon: "08", deep: false,
  title: "How Everything Connects",
  html: `
    <p class="lede">One solver at the bottom, three independent extensions around it, and
    concrete applications on top. Every arrow below is a real Python import — verified
    while doing the reorganization, not a conceptual diagram.</p>

    <div class="arch-diagram">
      <div class="arch-row">
        <div class="arch-box arch-app">todorov_toy<span>Todorov</span></div>
        <div class="arch-box arch-app">pendulum_constrained<span>Zahid</span></div>
        <div class="arch-box arch-app arch-highlight">sci_rehab<span>Katie — the deliverable</span></div>
        <div class="arch-box arch-ghost">(none yet)<span>Andrew's application</span></div>
      </div>
      <div class="arch-label">applications/</div>
      <div class="arch-row">
        <div class="arch-box arch-ext">constraints<span>Zahid</span></div>
        <div class="arch-box arch-ext">dual_control<span>Andrew</span></div>
        <div class="arch-box arch-ext">sindy<span>Katie</span></div>
      </div>
      <div class="arch-label">extensions/</div>
      <div class="arch-row">
        <div class="arch-box arch-core" style="grid-column: span 3">ddp_solver<span>Todorov</span></div>
      </div>
      <div class="arch-label">core/</div>
    </div>

    <h3>The real import edges</h3>
    <div class="connects">
      <div class="connect-row"><span class="arrow">todorov_toy &rarr; core</span>
        <p>Runs the plain solver on a linear system. No extension involved.</p></div>
      <div class="connect-row"><span class="arrow">pendulum_constrained &rarr; core, constraints</span>
        <p>Drives the raw solver directly and layers Zahid's constraint bounds on top.</p></div>
      <div class="connect-row"><span class="arrow">sci_rehab &rarr; dual_control, sindy</span>
        <p>Reuses Andrew's MPC loop completely unmodified, and Katie's own SINDy pipeline
        to identify the SCI plant before controlling it.</p></div>
      <div class="connect-row"><span class="arrow">dual_control &rarr; core</span>
        <p>Only <code>box_qp.py</code> is literally shared — the rest of Andrew's solver is
        an independent reimplementation, not a wrapper.</p></div>
      <div class="connect-row"><span class="arrow">constraints &rarr; dual_control</span>
        <p>Reuses <code>finite_difference.py</code> for numeric Jacobians.</p></div>
      <div class="connect-row flag"><span class="arrow">sindy &rarr; sci_rehab (reverse!)</span>
        <p><b>The one flagged wrinkle:</b> <code>validate_sindy_model.py</code> imports a
        helper (<code>build_sindy_row</code>) from the SCI plant to avoid a third copy of
        the same function. An extension depending on a specific application inverts the
        intended layering — kept and documented, not silently duplicated away.</p></div>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Core = foundation everyone can use. Extensions = reusable capabilities.
        Applications = concrete plant + demo.</li>
        <li>Andrew's dual_control is NOT a thin wrapper on core — mostly independent,
        sharing only box_qp.py.</li>
        <li>One deliberate exception to clean layering exists and is documented:
        sindy &rarr; sci_rehab.</li>
      </ol>
      <div class="one-liner">"Applications sit on extensions which sit on core, except for
      one documented exception where the SINDy extension reaches back up into Katie's
      SCI application to reuse one helper function."</div>
    </div>
  `
},

{
  id: "architecture", nav: "Architecture", icon: "09", deep: false,
  title: "Final Architecture — The Folder Structure",
  html: `
    <p class="lede">Reorganized 2026-08-17 from four flat, per-person folders
    (<code>PYTHON/todorov/</code>, <code>andrew/</code>, <code>katie/</code>,
    <code>zahid/</code> — now deleted) into this:</p>

    <pre class="tree">PYTHON/
├── core/
│   └── ddp_solver/            <span class="tree-note">Todorov — box_qp, back_pass, forward_pass, ilqg</span>
├── extensions/
│   ├── dual_control/          <span class="tree-note">Andrew — backward_pass, forward_pass, ilqg_function,
│   │                            todorov_estimator, spkf_function, simulate_system,
│   │                            make_pd, finite_difference, main_outer_control_loop</span>
│   ├── sindy/                 <span class="tree-note">Katie — the 7-file SINDy pipeline</span>
│   └── constraints/            <span class="tree-note">Zahid — dynamic_control_bounds, relative_degree_reduction</span>
└── applications/
    ├── sci_rehab/              <span class="tree-note">Katie — SCI plant, personas, run_katie_pipeline.py</span>
    ├── pendulum_constrained/  <span class="tree-note">Zahid — run_zahid_demo.py</span>
    └── todorov_toy/            <span class="tree-note">Todorov — run_todorov_demo.py</span></pre>

    <h3>Why this shape</h3>
    <ul class="reason-list">
      <li>The supervisor asked for one framework where the shared control algorithm is
      one core, and each person's contribution builds on it as a clear, separate layer —
      not merged into one file.</li>
      <li>Every directory is a real Python package (<code>__init__.py</code>); every
      cross-module import is a real absolute package import
      (e.g. <code>from core.ddp_solver.ilqg import ilqg</code>) — no more
      <code>sys.path.append(...)</code> hacks.</li>
      <li><code>PYTHON/pyproject.toml</code> + <code>pip install -e .</code> (run once)
      makes the whole thing importable from anywhere, so existing "run directly:
      <code>python run_katie_demo.py</code>" instructions still work unchanged.</li>
    </ul>

    <div class="callout beginner">
      <div class="callout-label">What didn't move</div>
      <p><code>CODES/</code> (the original MATLAB source trees) and <code>NOTES/</code>
      (diagrams) were left untouched — they're reference material, not part of the Python
      package tree.</p>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>10 packages total: 1 core, 3 extensions, 4 applications (only 3 populated —
        Andrew has none), plus supporting files.</li>
        <li>Real Python packages now, not a folder-of-scripts with path hacks.</li>
        <li>Old MATLAB source and diagrams were intentionally left alone.</li>
      </ol>
      <div class="one-liner">"The final structure is core/extensions/applications, with
      every directory a real Python package and every import a real package import — I
      moved every file and rewrote every import by hand, checkpointing tests after each
      group."</div>
    </div>
  `
},

{
  id: "flow", nav: "Follow the Data", icon: "10", deep: true,
  title: "Follow the Data — How the Code Actually Runs",
  html: `
    <p class="lede">This traces the <b>real</b> function calls inside
    <code>applications/sci_rehab/run_katie_pipeline.py</code> — the actual top-level script
    that runs the whole system end to end for one persona. Function names below
    (<code>run_identification</code>, <code>select_uncertain_terms</code>,
    <code>run_dual_control_planning</code>, <code>print_control_report</code>) are taken
    directly from the file, not invented.</p>

    <div class="exec-trace">
      <div class="exec-step">
        <div class="exec-num">1</div>
        <div class="exec-body">
          <div class="exec-title">Input: a persona name</div>
          <div class="exec-detail">e.g. <code>"frail_elder"</code> — everything downstream
          is driven by this one string.</div>
        </div>
      </div>
      <div class="exec-connector">&darr;</div>
      <div class="exec-step">
        <div class="exec-num">2</div>
        <div class="exec-body">
          <div class="exec-title"><code>run_identification()</code> — PART A begins</div>
          <div class="exec-detail">Calls <code>generate_synthetic_dataset()</code> to
          simulate many patients of this persona over several sessions (this stands in for
          "real clinic data" — the ground truth for identification).</div>
        </div>
      </div>
      <div class="exec-connector">&darr;</div>
      <div class="exec-step">
        <div class="exec-num">3</div>
        <div class="exec-body">
          <div class="exec-title">The SINDy chain runs</div>
          <div class="exec-detail"><code>build_sindy_library()</code> &rarr;
          <code>cross_validate_lambda()</code> &rarr; <code>stlsq()</code> &rarr;
          <code>prune_sindy_bic()</code> &rarr; static-state equations zeroed &rarr;
          <code>validate_sindy_model()</code> &rarr; <code>compute_fisher_information()</code>
          &rarr; <code>print_identified_equations()</code>. Output: a sparse identified
          model (<code>Xi</code>) plus a Fisher-information-based uncertainty estimate
          (<code>Sigma</code>) for the key coefficients.</div>
        </div>
      </div>
      <div class="exec-connector">&darr;</div>
      <div class="exec-step">
        <div class="exec-num">4</div>
        <div class="exec-body">
          <div class="exec-title"><code>select_uncertain_terms()</code></div>
          <div class="exec-detail">For each dynamic state (SCIM, BBS), picks the top-K
          nonzero <i>action</i> coefficients (by magnitude) as "clinically decision-relevant
          unknowns" — how much does <i>this</i> therapy actually help <i>this</i> state.
          Their starting uncertainty (<code>cov_P</code>) is seeded straight from the
          Fisher-information <code>Sigma</code> computed in step 3.</div>
        </div>
      </div>
      <div class="exec-connector">&darr;</div>
      <div class="exec-step">
        <div class="exec-num">5</div>
        <div class="exec-body">
          <div class="exec-title"><code>run_dual_control_planning()</code> — PART B begins</div>
          <div class="exec-detail">Reuses <code>extensions/dual_control/
          main_outer_control_loop.py</code> — the same generic MPC loop Andrew wrote for
          the COVID testbed — <b>completely unmodified</b>, pointed at Katie's SCI plant
          (<code>discrete_state_dynamics</code>, <code>measurement</code>,
          <code>l_cost</code>, <code>continuous_state_dynamics</code>) with the just-
          identified <code>Xi</code> as the model.</div>
        </div>
      </div>
      <div class="exec-connector">&darr;</div>
      <div class="exec-step">
        <div class="exec-num">6</div>
        <div class="exec-body">
          <div class="exec-title">Inside the MPC loop, every session:</div>
          <div class="exec-detail"><code>ilqg_function()</code> plans a full trajectory
          from the current belief &rarr; only the <b>first</b> therapy dose is applied
          &rarr; <code>simulate_system()</code> steps the (simulated) true patient forward
          one week &rarr; <code>spkf_function()</code> fuses the new observation to update
          both the state estimate <i>and</i> the belief about the uncertain coefficients
          from step 4 &rarr; warm-start &rarr; repeat next session.</div>
        </div>
      </div>
      <div class="exec-connector">&darr;</div>
      <div class="exec-step">
        <div class="exec-num">7</div>
        <div class="exec-body">
          <div class="exec-title"><code>print_control_report()</code> — final output</div>
          <div class="exec-detail">A readable report of the session-by-session therapy
          doses chosen, the resulting SCIM/BBS trajectory, and how the belief about the
          uncertain coefficients moved (dual control's own "did I actually learn
          something" story — mirroring the same true-vs-estimated check done for Andrew's
          original demo).</div>
        </div>
      </div>
    </div>

    <div class="callout beginner">
      <div class="callout-label">Where SINDy enters, where DDP enters</div>
      <p>SINDy enters once, up front, in PART A — it turns "a pile of simulated patient
      data" into "a sparse model + an uncertainty estimate." DDP/iLQG (via Andrew's
      dual-control wrapper) enters in PART B, and runs <i>repeatedly</i>, once per
      simulated clinic session, using that model and gradually refining the uncertain
      parts of it as therapy is delivered.</p>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Two parts: identify once (SINDy), then control repeatedly, session by session
        (dual iLQG via Andrew's MPC loop).</li>
        <li>The link between them is the Fisher-information covariance, which seeds how
        uncertain the controller starts out about the few coefficients it's allowed to
        keep learning online.</li>
        <li>main_outer_control_loop.py is reused byte-for-byte from Andrew's COVID work —
        it never needed to know it was now planning therapy instead of epidemic
        response.</li>
      </ol>
      <div class="one-liner">"The pipeline identifies a sparse model once with SINDy, then
      hands its uncertainty estimate straight into Andrew's unmodified dual-control MPC
      loop, which re-plans therapy every session and keeps refining the few coefficients
      it was told to treat as uncertain."</div>
    </div>
  `
},

{
  id: "modules", nav: "Module Reference", icon: "11", deep: true,
  title: "Module Reference — Input / Processing / Output",
  html: `
    <p class="lede">The most important files, what they take in, what they do, what they
    hand back, and who actually calls them (verified from the real import statements
    during the reorganization).</p>

    <div class="mod-card">
      <div class="mod-file">core/ddp_solver/ilqg.py</div>
      <div class="mod-why">Why it exists: one function you call to get a full optimized
      trajectory — hides the iterate-until-converged loop from callers.</div>
      <div class="mod-io">
        <div><b>Input</b> — step function, derivatives function, initial state, initial
        control guess, optional control limits/constraint function.</div>
        <div><b>Processing</b> — repeats backward pass &rarr; forward pass &rarr;
        line search until cost stops improving.</div>
        <div><b>Output</b> — optimized state/control trajectory, feedback gains, cost.</div>
      </div>
      <div class="mod-caller">Called by: applications/todorov_toy/run_todorov_demo.py,
      applications/pendulum_constrained/run_zahid_demo.py</div>
    </div>

    <div class="mod-card">
      <div class="mod-file">extensions/dual_control/ilqg_function.py</div>
      <div class="mod-why">Why it exists: the stochastic/estimation-aware equivalent of
      ilqg.py — the version that actually gets used for anything uncertain.</div>
      <div class="mod-io">
        <div><b>Input</b> — same as ilqg.py plus noise statistics and the current
        parameter-estimate covariance.</div>
        <div><b>Processing</b> — backward_pass.py + forward_pass.py (this package's own
        versions) + todorov_estimator.py.</div>
        <div><b>Output</b> — planned trajectory accounting for uncertainty.</div>
      </div>
      <div class="mod-caller">Called by: main_outer_control_loop.py,
      applications/sci_rehab/run_katie_demo.py</div>
    </div>

    <div class="mod-card">
      <div class="mod-file">extensions/dual_control/main_outer_control_loop.py</div>
      <div class="mod-why">Why it exists: the session-by-session MPC loop — written
      generically (plant passed in as arguments) specifically so it could be reused for a
      completely different plant later without any changes.</div>
      <div class="mod-io">
        <div><b>Input</b> — dynamics/measurement/cost functions, initial belief, number of
        sessions.</div>
        <div><b>Processing</b> — plan (ilqg_function) &rarr; step real system
        (simulate_system) &rarr; re-estimate (spkf_function) &rarr; warm-start &rarr;
        repeat.</div>
        <div><b>Output</b> — full history of applied actions, true/estimated states,
        parameter belief over time.</div>
      </div>
      <div class="mod-caller">Called by: applications/sci_rehab/run_katie_pipeline.py
      — <i>completely unmodified</i>, just pointed at a different plant.</div>
    </div>

    <div class="mod-card">
      <div class="mod-file">extensions/sindy/build_sindy_library.py</div>
      <div class="mod-why">Why it exists: turns raw patient trajectories into the
      regression problem SINDy actually solves.</div>
      <div class="mod-io">
        <div><b>Input</b> — patient state/action trajectories (X, A), a config
        (polynomial order, cross-terms on/off, protected terms).</div>
        <div><b>Processing</b> — finite-differences the states to get dX; builds the
        candidate-term table (Theta) at every timestep; column-normalizes.</div>
        <div><b>Output</b> — Theta (candidate library), dX (targets), column-normalization
        info, term labels.</div>
      </div>
      <div class="mod-caller">Called by: cross_validate_lambda.py, stlsq.py (indirectly, via
      the pipeline script), applications/sci_rehab/run_katie_pipeline.py,
      applications/sci_rehab/generate_synthetic_dataset.py</div>
    </div>

    <div class="mod-card">
      <div class="mod-file">extensions/sindy/stlsq.py</div>
      <div class="mod-why">Why it exists: the actual sparse solve — the part that turns
      "200 candidate terms" into "a handful of real ones."</div>
      <div class="mod-io">
        <div><b>Input</b> — Theta, dX, sparsity threshold (lambda), protected-term
        indices, min/max term counts.</div>
        <div><b>Processing</b> — alternates ridge-regression refits with hard-thresholding
        of small normalized coefficients until the active-term set stabilizes.</div>
        <div><b>Output</b> — Xi, the sparse coefficient matrix (mostly zeros).</div>
      </div>
      <div class="mod-caller">Called by: cross_validate_lambda.py (during the CV grid
      search), applications/sci_rehab/run_katie_pipeline.py (final fit)</div>
    </div>

    <div class="mod-card">
      <div class="mod-file">applications/sci_rehab/continuous_state_dynamics.py</div>
      <div class="mod-why">Why it exists: the actual SCI physiology model — replaces
      Andrew's COVID equations entirely.</div>
      <div class="mod-io">
        <div><b>Input</b> — current state (6), current therapy doses (13), model
        coefficients (identified Xi or population-average placeholder).</div>
        <div><b>Processing</b> — computes instantaneous rate of change per state, with a
        recovery-ceiling clamp so gains taper as a patient approaches their personal
        ceiling.</div>
        <div><b>Output</b> — state derivative vector.</div>
      </div>
      <div class="mod-caller">Called by: discrete_state_dynamics.py (every simulated
      timestep), and — the one flagged reverse dependency — imported by
      extensions/sindy/validate_sindy_model.py for its build_sindy_row helper.</div>
    </div>

    <div class="mod-card">
      <div class="mod-file">applications/sci_rehab/l_cost.py</div>
      <div class="mod-why">Why it exists: defines "good" for this specific problem —
      functional recovery vs. therapy burden, weighted per-persona.</div>
      <div class="mod-io">
        <div><b>Input</b> — state, action (doses), persona cost weights.</div>
        <div><b>Processing</b> — rewards SCIM/BBS gains, penalizes total dose/resource use
        by the persona's own burden-sensitivity weights.</div>
        <div><b>Output</b> — a single cost number (plus derivatives, for the solver).</div>
      </div>
      <div class="mod-caller">Called by: every planning step inside ilqg_function.py, via
      main_outer_control_loop.py</div>
    </div>

    <div class="mod-card">
      <div class="mod-file">extensions/constraints/dynamic_control_bounds.py</div>
      <div class="mod-why">Why it exists: turns "this quantity must stay above/below X"
      into "here's the tightened control range right now," which the existing box-QP
      solver already knows how to use.</div>
      <div class="mod-io">
        <div><b>Input</b> — a constraint function, the current trajectory point, optional
        analytic Jacobian (falls back to numeric).</div>
        <div><b>Processing</b> — first-order solve for the control bound implied by the
        constraint at this exact point, intersected with any hard limits.</div>
        <div><b>Output</b> — a per-timestep (upper, lower) control bound array.</div>
      </div>
      <div class="mod-caller">Called by: applications/pendulum_constrained/run_zahid_demo.py,
      via core/ddp_solver/ilqg.py's optional constraint_fn hook</div>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Every important module was already independently tested before being wired
        into the pipeline — that's why the end-to-end script could be built with high
        confidence.</li>
        <li>"Who calls it" was verified directly from the actual import statements while
        doing the reorganization, not guessed.</li>
        <li>main_outer_control_loop.py is the clearest example of good design here — it
        never had to change to serve a completely different application.</li>
      </ol>
      <div class="one-liner">"I can point to exactly which function calls which, in the
      real code, because I personally rewired every import during the reorganization and
      re-verified the whole suite after each step."</div>
    </div>
  `
},

{
  id: "contribution", nav: "My Contribution", icon: "12", deep: false,
  title: "What Did I Actually Contribute?",
  html: `
    <p class="lede">Separating "inherited from the original researchers" from
    "built or done by me" — this is the question a professor is most likely to probe.</p>

    <div class="contrib-grid">
      <div class="contrib-col">
        <div class="contrib-head">Inherited (the researchers' ideas/math)</div>
        <ul>
          <li>The DDP/iLQG algorithm itself (Tassa, Mansard &amp; Todorov, 2014).</li>
          <li>The dual-control / SPKF estimation design (Andrew).</li>
          <li>The SCI state space, WHO therapy list, persona definitions, and SINDy
          pipeline design (Katie).</li>
          <li>The constrained-DDP method and its equations (Dastan &amp; Sensinger's
          paper, Zahid).</li>
        </ul>
      </div>
      <div class="contrib-col mine">
        <div class="contrib-head">Actually done by me</div>
        <ul>
          <li>Ported Todorov's, Andrew's, and Katie's MATLAB code to Python, file by file,
          each verified with new tests before moving on.</li>
          <li>Built Zahid's method from scratch in Python directly from his paper's
          equations — no source existed anywhere to port from.</li>
          <li>Found and fixed a real latent symmetry bug in the original MATLAB
          (backward_pass.m), invisible there only due to how MATLAB reads matrices.</li>
          <li>Found and fixed multiple MATLAB-vs-NumPy numerical-convention traps
          (std ddof, NaN handling in max) that would have silently corrupted the SINDy
          pipeline's thresholding.</li>
          <li>Root-caused a real convergence failure in Zahid's method at the paper's own
          stated parameter, confirmed it wasn't a bug, and found a parameter value that
          honestly reproduces the paper's result.</li>
          <li>Designed and executed the full core/extensions/applications
          reorganization: moved every file, converted every cross-file import from a
          sys.path hack to a real Python package import, and re-verified the entire test
          suite after every group of changes — zero regressions, 132/132.</li>
          <li>Found and corrected a real staleness in the project's own documentation
          (CLAUDE.md claimed a script was "still needed" that had, in fact, already been
          built).</li>
          <li>Wrote every automated test in the project — deterministic closed-form checks
          where possible (LQR, hand-derived pendulum), behavioral checks where a
          closed-form doesn't exist (dual-control convergence).</li>
        </ul>
      </div>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Three ports (Todorov, Andrew, Katie) + one from-scratch implementation
        (Zahid), all independently tested.</li>
        <li>Several real bugs found — not invented problems, genuine latent issues in the
        original MATLAB or in the MATLAB-to-Python conversion conventions.</li>
        <li>The reorganization itself — architecture design, execution, and verification
        — is entirely my own work on top of the four conversions.</li>
      </ol>
      <div class="one-liner">"I didn't invent the control theory or the SCI model — I
      converted and verified three MATLAB codebases and built a fourth from a paper with
      no code, found several real bugs along the way, and then designed and executed the
      reorganization that turned four disconnected folders into one tested framework."</div>
    </div>
  `
},

{
  id: "findings", nav: "Findings", icon: "13", deep: false,
  title: "Final Findings",
  html: `
    <p class="lede">The findings worth remembering, each as Expected &rarr; Observed
    &rarr; Why it matters.</p>

    <div class="finding-card">
      <div class="finding-title">1. Latent symmetry bug in Todorov's backward pass</div>
      <div class="finding-part"><span class="finding-label">Expected</span>
      A particular matrix in the backward-pass math (Sxxh) should be symmetric.</div>
      <div class="finding-part"><span class="finding-label">Observed</span>
      The original MATLAB computed it asymmetrically — never caught because MATLAB's
      Cholesky solver silently only reads one triangle.</div>
      <div class="finding-part"><span class="finding-label">Why it matters</span>
      A real, pre-existing bug worth telling the supervisor about. Fixed in the Python
      port by explicitly symmetrizing.</div>
    </div>

    <div class="finding-card">
      <div class="finding-title">2. MATLAB/NumPy standard-deviation convention mismatch</div>
      <div class="finding-part"><span class="finding-label">Expected</span>
      A direct MATLAB-to-Python port of <code>std()</code> should behave identically.</div>
      <div class="finding-part"><span class="finding-label">Observed</span>
      MATLAB's default divides by N-1; NumPy's default divides by N — a scale error of
      &radic;(N/(N-1)) if not handled.</div>
      <div class="finding-part"><span class="finding-label">Why it matters</span>
      Would have silently mis-scaled every SINDy thresholding decision across five files.
      Fixed with explicit <code>ddof=1</code> everywhere it appears.</div>
    </div>

    <div class="finding-card">
      <div class="finding-title">3. Andrew's dual control genuinely learns, not just runs</div>
      <div class="finding-part"><span class="finding-label">Expected</span>
      If dual control works, a deliberately wrong parameter guess should self-correct
      while the controller keeps acting.</div>
      <div class="finding-part"><span class="finding-label">Observed</span>
      Exactly that — a guess of 0.4 converges toward the true value of 1.0 while cost
      drops over iterations.</div>
      <div class="finding-part"><span class="finding-label">Why it matters</span>
      Confirms the dual-control mechanism is doing real work, not a trivial pass-through
      — the strongest test available given there's no closed-form ground truth here.</div>
    </div>

    <div class="finding-card">
      <div class="finding-title">4. Zahid's paper alpha traps the solver</div>
      <div class="finding-part"><span class="finding-label">Expected</span>
      The paper's own &alpha;=0.1 should converge to its reported cost (~10.11) in about
      16 iterations.</div>
      <div class="finding-part"><span class="finding-label">Observed</span>
      It gets stuck after one accepted step (cost stuck at 149.6); &alpha;=1.0 with the
      identical code converges cleanly (cost 11.46) and honestly matches the paper's
      qualitative figure.</div>
      <div class="finding-part"><span class="finding-label">Why it matters</span>
      Root-caused, not a bug — the allowed control range is genuinely razor-thin early in
      the swing at small &alpha;. A real, disclosed limitation of the method at its
      literal stated parameter.</div>
    </div>

    <div class="finding-card">
      <div class="finding-title">5. Reorganization fixed a real testing limitation</div>
      <div class="finding-part"><span class="finding-label">Expected</span>
      The whole project's tests should be runnable as one command.</div>
      <div class="finding-part"><span class="finding-label">Observed</span>
      Under the old flat layout, two files named <code>forward_pass.py</code> in different
      folders collided when Python tried to run the whole tree in one pytest invocation —
      testing had to be done per-directory.</div>
      <div class="finding-part"><span class="finding-label">Why it matters</span>
      The core/extensions/applications reorganization gives every module a fully-qualified
      package name, which eliminates the collision — the entire 132-test suite now runs
      as a single command for the first time.</div>
    </div>

    <div class="finding-card">
      <div class="finding-title">6. Documentation had drifted from the real code</div>
      <div class="finding-part"><span class="finding-label">Expected</span>
      CLAUDE.md's "still needed" list should match what actually exists in the code.</div>
      <div class="finding-part"><span class="finding-label">Observed</span>
      It listed a top-level orchestration script as "still needed" that had, in fact,
      already been built and tested weeks earlier.</div>
      <div class="finding-part"><span class="finding-label">Why it matters</span>
      Caught and corrected during the reorganization — a reminder that project docs need
      periodic reconciliation against the actual files, not just narration at write-time.</div>
    </div>

    <div class="callout warn">
      <div class="callout-label">⚠️ Open discrepancy — not yet resolved</div>
      <p>CLAUDE.md's written status for Andrew's conversion says "24/24 tests passing,"
      but actually running the current suite gives 16/16. This has been flagged, not
      silently resolved either way — worth a quick check before quoting a number to the
      professor.</p>
    </div>
  `
},

{
  id: "status", nav: "Status", icon: "14", deep: false,
  title: "What Is Complete vs Incomplete",
  html: `
    <div class="status-list">
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Todorov's DDP/iLQG solver</b> — ported, verified against exact LQR
        solution.</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Andrew's dual-control solver core</b> — ported, verified behaviorally
        (parameter convergence).</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Katie's SCI plant/cost (4 files)</b> — ported and tested.</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Katie's SINDy pipeline (7 files)</b> — ported and tested, hand-port
        decision documented.</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Katie's personas (18) and WHO therapies (13)</b> — ported and tested.</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Katie's synthetic-data generator</b> — done and tested.</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Katie's top-level orchestration</b> (<code>run_katie_pipeline.py</code> +
        preview) — exists, wires the entire chain end to end for one persona at a time.</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>Zahid's constraint extension</b> (both variants, wired into the solver,
        demo) — feature-complete relative to the plan; verification is a qualitative
        paper-match, by nature, since no original code exists to diff against.</div></div>
      <div class="status-row"><span class="status-dot green"></span>
        <div><b>core/extensions/applications reorganization</b> — done, 132/132 tests,
        CLAUDE.md updated to match.</div></div>
      <div class="status-row"><span class="status-dot yellow"></span>
        <div><b>Andrew test-count documentation</b> — CLAUDE.md says 24/24, live suite
        says 16/16. Flagged, not resolved.</div></div>
      <div class="status-row"><span class="status-dot red"></span>
        <div><b>Andrew's SIDARTHE COVID plant, ported to Python</b> — never done. Open
        question for the supervisor: keep COVID dynamics for that branch, or treat
        SCI-only as the answer in practice (Katie's code already replaced it there)?</div></div>
      <div class="status-row"><span class="status-dot red"></span>
        <div><b>Full multi-persona / cohort-scale orchestration</b>
        (a Python equivalent of <code>run_sgba_personas.m</code>, cohort sensitivity, the
        paper-results scripts) — not built. What exists (<code>run_katie_pipeline.py</code>)
        runs one persona at a time, demonstrated on 2 of the 18.</div></div>
    </div>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>Every conversion (Todorov, Andrew, Katie, Zahid) and the reorganization itself
        are done and tested.</li>
        <li>Two real open items: Andrew's SIDARTHE plant was never ported (a supervisor
        decision point), and there's no full 18-persona cohort sweep yet — only
        single-persona runs.</li>
        <li>One small documentation inconsistency (Andrew's test count) is flagged, not
        yet resolved.</li>
      </ol>
      <div class="one-liner">"Everything asked for so far is done and tested; the two
      honest gaps are Andrew's COVID plant, which was never ported, and a full
      18-persona cohort run, which doesn't exist yet — only single-persona runs do."</div>
    </div>
  `
},

{
  id: "numbers", nav: "Numbers", icon: "15", deep: false,
  title: "Important Numbers to Remember",
  html: `
    <div class="num-grid">
      <div class="num-tile"><div class="num-val">132/132</div><div class="num-label">tests
      passing, whole project, single run</div></div>
      <div class="num-tile"><div class="num-val">4</div><div class="num-label">original
      contributors/codebases</div></div>
      <div class="num-tile"><div class="num-val">3</div><div class="num-label">framework
      layers: core, extensions, applications</div></div>
      <div class="num-tile"><div class="num-val">6</div><div class="num-label">core
      solver tests (Todorov)</div></div>
      <div class="num-tile"><div class="num-val">16</div><div class="num-label">dual-control
      tests (Andrew) — live count; docs say 24, flagged</div></div>
      <div class="num-tile"><div class="num-val">31</div><div class="num-label">SINDy
      pipeline tests (Katie, 7 files)</div></div>
      <div class="num-tile"><div class="num-val">31</div><div class="num-label">constraint
      extension tests (Zahid, 2 files)</div></div>
      <div class="num-tile"><div class="num-val">48</div><div class="num-label">SCI
      application tests (Katie's plant/personas/pipeline)</div></div>
      <div class="num-tile"><div class="num-val">6</div><div class="num-label">SCI state
      variables: SCIM, BBS, AIS, Age, DPI, Caregiver support</div></div>
      <div class="num-tile"><div class="num-val">13</div><div class="num-label">WHO
      therapies (the action space)</div></div>
      <div class="num-tile"><div class="num-val">18</div><div class="num-label">patient
      personas (5 archetypes + 12 named + 1 default)</div></div>
      <div class="num-tile"><div class="num-val">2026-08-17</div><div class="num-label">reorganization
      date</div></div>
    </div>
    <p class="lede" style="margin-top:8px">6 + 16 + 31 + 31 + 48 = 132 — that's the whole
    suite, package by package, verified by actually running pytest during the
    reorganization.</p>

    <div class="remember-box">
      <div class="remember-title">What you should remember</div>
      <ol>
        <li>132/132 is the one number to lead with if asked "does it work?"</li>
        <li>6 states, 13 actions, 18 personas describes the SCI problem size in one
        breath.</li>
        <li>The 132 breaks down cleanly by package: 6+16+31+31+48.</li>
      </ol>
      <div class="one-liner">"132 out of 132 tests pass across the whole framework, split
      6/16/31/31/48 across the five tested packages."</div>
    </div>
  `
},

{
  id: "questions", nav: "Professor Q&A", icon: "16", deep: false,
  title: "Likely Professor Questions",
  html: `
    <p class="lede">Click a question to reveal a short, sayable answer.</p>
    <div class="qa-list" id="qa-list"></div>
  `
},

{
  id: "presentation", nav: "Presentation", icon: "17", deep: false,
  title: "Presentation Preparation",
  html: `
    <h3>30-second version</h3>
    <div class="script-card">
      <p>"My Mitacs project applies curiosity-driven AI — controlling a system while
      learning about it at the same time — to spinal cord injury rehab. I took four
      separate codebases — a generic optimal-control solver, an uncertainty-aware
      extension of it, a real SCI patient model with data-driven system identification,
      and a hard-safety-constraint extension — ported and verified all of them in Python,
      132 out of 132 tests passing, and reorganized them into one shared framework."</p>
    </div>

    <h3>2-minute version</h3>
    <div class="script-card">
      <p>"The project has four layers. At the bottom is Todorov's DDP/iLQG solver — a
      generic algorithm that plans the best sequence of actions for any system you can
      describe with states, controls, dynamics, and a cost function. Andrew extended that
      to handle uncertainty: his 'dual control' approach plans actions while
      simultaneously estimating unknown system parameters from a Kalman-filter-style
      estimator, tested originally on a COVID epidemic model. Katie's contribution
      replaces that COVID model with a real SCI patient model — six clinical variables,
      13 WHO-defined therapies, 18 patient personas — and adds SINDy, which discovers the
      governing equations directly from simulated patient data instead of assuming them,
      producing a sparse, clinician-readable model. Zahid's work is separate: a paper-only
      method, which I implemented from scratch since no code existed, that lets the
      solver enforce hard safety constraints.

      For the final phase, I reorganized all four into one framework — core, extensions,
      applications — real Python packages, real imports, zero test regressions, 132 out
      of 132 tests passing in a single run. Along the way I found and fixed a real bug in
      the original MATLAB solver, several MATLAB-to-Python numerical-convention traps in
      the SINDy code, and root-caused a genuine convergence limitation in Zahid's
      method at the paper's own stated parameter."</p>
    </div>

    <h3>5-minute version</h3>
    <div class="script-card">
      <p>Use the 2-minute version as the spine, then add, in order:</p>
      <ol>
        <li><b>Concrete example</b>: the two-persona demo — "default" front-loads therapy
        and favors balance/gait/fitness training; "frail_elder" doses conservatively and
        prefers a different therapy mix, because the difference is baked into the
        simulated physiology, not hand-coded rules.</li>
        <li><b>Walk the pipeline</b>: identify once with SINDy (7 stages, ending in a
        Fisher-information uncertainty estimate) &rarr; control repeatedly with Andrew's
        unmodified MPC loop, refining just the few clinically-important uncertain
        coefficients each session.</li>
        <li><b>The Zahid finding</b>: the paper's own alpha value traps the solver — root
        caused, not a bug, and alpha=1.0 honestly reproduces the paper's figure.</li>
        <li><b>Close with the two honest gaps</b>: Andrew's COVID plant was never ported
        (open question for you), and there's no full 18-persona cohort sweep yet, only
        single-persona runs.</li>
      </ol>
    </div>

    <h3>Numbers to have ready</h3>
    <p>132/132 tests &middot; 4 contributors &middot; 3 layers &middot; 6 SCI states
    &middot; 13 therapies &middot; 18 personas &middot; 2026-08-17 reorg date.</p>

    <div class="remember-box">
      <div class="remember-title">One sentence, if you only get one</div>
      <div class="one-liner">"I converted three MATLAB control-theory codebases and built
      a fourth from a paper with no code, found and fixed several real bugs along the
      way, and organized all four into one tested Python framework — 132 out of 132 tests
      passing."</div>
    </div>
  `
},

];

/* ---- Professor Q&A data (separate from chapters so app.js can render as an accordion) ---- */
const QA = [
  {
    q: "What exactly did YOU do, versus what did you inherit from the original researchers?",
    a: "I ported Todorov's, Andrew's, and Katie's MATLAB code to Python with new tests for each; Zahid had no code at all, so I implemented his method from his paper's equations directly. On top of that, I found and fixed a real bug in the original MATLAB solver, found several MATLAB/NumPy numerical-convention traps in the SINDy code, root-caused a real convergence issue in Zahid's method, and designed and executed the full reorganization into one framework — that reorganization is entirely my own architectural work, not inherited from anyone."
  },
  {
    q: "Why convert this to Python at all?",
    a: "Honestly, the specific reason wasn't recorded in my project notes as a stated rationale — worth confirming with your supervisor directly if asked. A reasonable answer: Python's open ecosystem (numpy/scipy/scikit-learn/matplotlib, no license cost) makes it easier to test rigorously, integrate with future curiosity-driven RL work, and share/reproduce results outside a MATLAB license."
  },
  {
    q: "What is DDP/iLQG, in one sentence?",
    a: "It's an algorithm that starts from a rough guess at a sequence of actions and repeatedly improves it — backward pass computes a local correction, forward pass applies it — until it converges on a locally-optimal plan."
  },
  {
    q: "What is dual control, in one sentence?",
    a: "Controlling a system and learning its unknown parameters at the same time, instead of learning first and controlling second."
  },
  {
    q: "What is SINDy, and why didn't you just use the pysindy package?",
    a: "SINDy discovers a sparse set of governing equations directly from data. The pysindy package doesn't support the specific custom logic this project depends on — protected terms, min/max term counts, custom cross-validation, BIC-based pruning — and normalizes columns differently, which would have broken verification against the MATLAB source. So I hand-ported it, same as everything else."
  },
  {
    q: "What's the latent bug you found in the original MATLAB code?",
    a: "In the backward pass, a matrix that must mathematically be symmetric was computed asymmetrically. It never caused a problem in MATLAB because its Cholesky solver silently only reads one triangle of the matrix — but it's a real, pre-existing bug. I fixed it in the Python port by symmetrizing the matrix explicitly."
  },
  {
    q: "Why doesn't Zahid's implementation match the paper's numbers exactly?",
    a: "There was no original code to port from or verify against — only the paper's equations. I verified against the paper's own hand-derived closed-form example and a qualitative match to its reported figure, not bit-exact numeric parity like the other three conversions. That's a disclosed limitation, not an oversight."
  },
  {
    q: "What would you do next if you had more time?",
    a: "Three things: get a supervisor decision on whether to port Andrew's original SIDARTHE COVID plant or commit fully to the SCI-only path; build the full multi-persona cohort orchestration (all 18 personas, not just the two demonstrated); and resolve the one open documentation discrepancy around Andrew's exact test count."
  },
  {
    q: "How do you know your Python port is actually correct, not just 'runs without crashing'?",
    a: "Each piece was verified the strongest way available for it: Todorov's solver against an independently-derived closed-form LQR solution (exact match); Zahid's method against the paper's own hand-derived closed form; Andrew's dual control behaviorally, by confirming a deliberately wrong parameter guess converges to the true value while cost drops; and everything with a MATLAB source was checked file-by-file against that source's own output, which is how I caught the ddof and NaN-handling numerical traps."
  },
  {
    q: "Why is the folder structure organized as core / extensions / applications?",
    a: "My supervisor asked for one framework where the shared algorithm is a single core, and each person's contribution is a clear, separate layer on top of it, rather than everything merged into one file. Core is Todorov's generic solver; extensions are the three add-on capabilities (dual control, SINDy, constraints); applications are the concrete plants and demos on top."
  },
  {
    q: "What's the actual clinical takeaway from the SCI demo?",
    a: "That the controller's recommendations genuinely differ per patient because persona differences are baked into the simulated physiology, not hand-coded rules — for example, a frail, burden-averse persona doses more conservatively and prefers different therapies than the default persona. Important caveat: this is a research prototype on simulated data, not validated on real patients."
  },
  {
    q: "Is there anything you're still unsure about or haven't finished?",
    a: "Yes, and I'd rather say so directly: Andrew's original SIDARTHE COVID plant was never ported to Python, so his codebase has no Python 'application' layer yet — that's an open decision for my supervisor. There's also no full 18-persona cohort-scale run yet, only single-persona runs. And there's one small unresolved documentation discrepancy about Andrew's exact test count."
  },
];
