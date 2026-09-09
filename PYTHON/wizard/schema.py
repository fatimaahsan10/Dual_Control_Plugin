"""
Generic, domain-agnostic model configuration for the wizard.

This is the ENTIRE answer set the question flow collects, and the ENTIRE
input to code generation (generic_plant.py) -- nothing else flows between
the two. It deliberately uses only generic vocabulary (states, actions,
parameters, constants, dynamics, cost, measurement) so the same dataclasses
describe a single-state/single-unknown case (dynamic_pricing/'s shape) or a
multi-state/multi-unknown case with cross-terms (duopoly_pricing/'s shape)
identically -- nothing here is aware of what a state/action/parameter is
FOR.

Persistence: to_dict()/from_dict() round-trip through plain JSON (see
storage.py) -- every field is a JSON-safe primitive, list, or nested dict,
never a numpy array or callable.

Two independent validation layers, deliberately kept separate:
  - ModelConfig.validate() (this file): structural sanity that doesn't
    require parsing any equation -- name uniqueness, shape/count
    consistency, numeric ranges, hard caps on solver settings that bound
    worst-case runtime. Cheap, always run first.
  - equation_parser.py: actually parses the dynamics/measurement/cost
    expression strings, which requires knowing the full symbol whitelist
    this file's fields imply. Kept in a separate module because it has a
    different failure mode (sympy exceptions) and is reused standalone by
    the wizard's live-validate-as-you-type equation inputs.

Neither layer touches core/ddp_solver/protocols.py or
extensions/dual_control/protocols.py -- this is a NEW, generic
configuration format that generic_plant.py later turns into callables
satisfying those (unmodified) contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

SCHEMA_VERSION = 1

# Which solver drives a model. "ilqg" (default, backward-compatible with
# every config saved before this field existed) routes through
# extensions/dual_control/main_outer_control_loop() exactly as today --
# MPC replanning + SPKF state/parameter estimation every step. "ilqr"
# routes through core/ddp_solver/ilqg.py instead: one deterministic
# full-horizon solve, no estimation, no replanning -- see
# wizard/core_ilqr_adapter.py. BOTH control methods can attach constraints
# (see ConstraintSpec below): main_outer_control_loop()/ilqg_function() now
# accept the same optional constraint_fn hook core/ddp_solver/ilqg.py
# already had -- see extensions/dual_control/ilqg_function.py's own
# docstring for the one real difference between the two paths (a
# "state_only" constraint's Lie-derivative reduction uses this model's
# fixed/prior parameter values even under "ilqg", never the online
# parameter ESTIMATE, since the reduction is built once from compile_plant()
# rather than re-built every session).
CONTROL_METHODS = ("ilqg", "ilqr")

# ConstraintSpec.kind values -- see ConstraintSpec's own docstring.
CONSTRAINT_KINDS = ("state_action", "state_only")

# ASCII-only identifier pattern, deliberately stricter than Python's own
# str.isidentifier() (which permits unicode letters) -- equation_parser.py's
# character whitelist (its first line of defense against sandbox-escape
# input, see that module's docstring) is ASCII-only, so every name accepted
# here must also be something equation_parser.py can safely reference.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Hard caps on anything that scales solver runtime, enforced in validate()
# regardless of what the UI widgets themselves already restrict -- a
# hand-edited JSON file must not be able to bypass these (see CLAUDE-adjacent
# plan §6: "never crash" / bounded worst-case runtime).
MAX_N_SESSIONS = 200
MAX_DU_ITERATIONS = 300
MAX_FIRST_RUN_DU_ITERATIONS = 500


@dataclass
class StateSpec:
    name: str
    label: str = ""
    initial_value: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    process_noise: float = 1.0  # dynamics noise scale for this state (diagonal of F)


@dataclass
class ActionSpec:
    name: str
    label: str = ""
    min: float = 0.0
    max: float = 1.0


@dataclass
class ParameterSpec:
    """An unknown quantity estimated online via dual control.

    `true_value` is optional and used ONLY for simulated demo runs (to
    check whether the estimate converges toward it) -- see
    dynamic_pricing/duopoly_pricing's own p_true convention. Leave None
    for a model meant to run against real data.
    """
    name: str
    label: str = ""
    prior_guess: float = 0.0
    prior_variance: float = 1.0
    process_noise: float = 0.05
    true_value: Optional[float] = None


@dataclass
class ConstantSpec:
    name: str
    value: float = 0.0


@dataclass
class MeasurementSpec:
    name: str
    expression: str
    noise_scale: float = 1.0


@dataclass
class CostSpec:
    running: str = "0"
    terminal: str = "0"


@dataclass
class ConstraintSpec:
    """One inequality constraint h(...) >= 0, enforced in EITHER
    control-method model (see CONTROL_METHODS above) via
    wizard/core_ilqr_adapter.py's build_constraint_fn() composing the
    EXISTING, UNMODIFIED extensions/constraints/ (Dastan & Sensinger 2024)
    machinery -- this schema field does not introduce any new constraint
    math of its own, only a generic way to describe which of that
    machinery's two variants a given expression should go through.
    build_constraint_fn()'s own output is control-method-agnostic (a plain
    callable(x_traj, u_traj) -> (N,m,2) array); wizard/solver_runner.py
    passes the SAME compiled constraint_fn to whichever solver
    config.control_method selects.

    kind == "state_action": `expression` is h(x, u) -- may reference
        states, actions, and constants. Passed directly to
        extensions/constraints/dynamic_control_bounds.py's Variant A.
        `alpha` is unused.
    kind == "state_only": `expression` is h(x) -- may reference states
        and constants ONLY (not actions -- enforced at parse time by
        giving equation_parser.py a narrower name whitelist for this
        kind, same mechanism schema.py already uses everywhere else).
        Reduced to a control-dependent constraint via
        extensions/constraints/relative_degree_reduction.py's Variant B
        (one Lie derivative + `alpha`-weighted decay, eq 18-19 of the
        paper) before being handed to Variant A.

    RELATIVE-DEGREE LIMITATION (inherited from the underlying
    implementation, not introduced here -- see
    extensions/constraints/dynamic_control_bounds.py's own docstring):
    a constraint only actually constrains anything if it (or, for
    kind=="state_only", its Lie derivative) depends on EXACTLY ONE
    action channel. A row that depends on zero channels is silently left
    unenforced (a RuntimeWarning is raised deep in that library code);
    the wizard surfaces this at Validate time rather than letting it pass
    silently -- see wizard/validation_runner.py. A row depending on TWO OR
    MORE channels raises a plain-language error at Validate/Run time
    instead of producing a wrong bound.
    """
    name: str
    kind: str = "state_only"
    expression: str = ""
    alpha: float = 1.0
    enabled: bool = True


@dataclass
class SolverSettings:
    """
    NOTE: augment_states_in_ilqg/augment_states_in_filter are
    deliberately NOT configurable fields here (unlike
    main_outer_control_loop()'s own parameters of the same name) --
    solver_runner.py always derives BOTH directly from whether the model
    has any unknown parameters (`n_p > 0`), never exposing them as an
    independent choice. Found while testing, in two parts:

    (1) When n_p > 0: augmenting must be True. main_outer_control_loop's
        `augment_states_in_filter=False` branch does
        `constants_filter = np.concatenate([constants, p_hat_0])`, which
        requires `constants` to be a raw ndarray -- incompatible with a
        custom `constants` OBJECT (ours, GenericConstants in
        generic_plant.py, and equally both real example plants'
        PricingConstants/DuopolyPricingConstants). Both
        dynamic_pricing/duopoly_pricing sidestep this the same way:
        hardcoding augment_states_in_filter=True unconditionally.

    (2) When n_p == 0: augmenting must be False, for the OPPOSITE reason
        -- main_outer_control_loop's augment_states_in_filter=True branch
        unconditionally does `Fp[0, 0]` when re-estimating each session
        (to broadcast a scalar parameter-noise value across the augmented
        block of the filter covariance), which raises IndexError when
        `Fp` is the (0, 0) matrix continuous_dynamics returns for zero
        parameters. Neither real example plant ever exercises this path
        (both always have n_p >= 1), so this is a previously-undiscovered
        edge case in main_outer_control_loop.py itself, not something
        introduced here -- and out of scope to fix (main_outer_control_loop.py
        is not modified by this project). solver_runner.py instead passes
        `constants` as a bare empty ndarray in this case (satisfying (1)'s
        concatenate requirement) rather than the GenericConstants object,
        which is safe here specifically because n_p=0 means neither
        generic_plant.py's dynamics/cost/measurement closures ever read
        from `constants` at all.
    """
    u_lim_method: int = 1
    reg_type: int = 1
    max_du_iterations: int = 60
    first_run_max_du_iterations: int = 100
    seed: int = 0


@dataclass
class ModelConfig:
    name: str
    description: str = ""
    dt: float = 1.0
    n_sessions: int = 15
    control_method: str = "ilqg"
    states: list[StateSpec] = field(default_factory=list)
    actions: list[ActionSpec] = field(default_factory=list)
    parameters: list[ParameterSpec] = field(default_factory=list)
    constants: list[ConstantSpec] = field(default_factory=list)
    dynamics: dict[str, str] = field(default_factory=dict)  # state name -> expression
    measurement: list[MeasurementSpec] = field(default_factory=list)
    cost: CostSpec = field(default_factory=CostSpec)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    solver: SolverSettings = field(default_factory=SolverSettings)
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Symbol-table helpers, shared with equation_parser.py so the
    # whitelist it checks against is always derived from THIS config,
    # never hand-duplicated.
    # ------------------------------------------------------------------

    def state_names(self) -> list[str]:
        return [s.name for s in self.states]

    def action_names(self) -> list[str]:
        return [a.name for a in self.actions]

    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def constant_names(self) -> list[str]:
        return [c.name for c in self.constants]

    def all_defined_names(self) -> list[str]:
        return (self.state_names() + self.action_names()
                + self.parameter_names() + self.constant_names())

    def constraint_names(self) -> list[str]:
        return [c.name for c in self.constraints]

    @property
    def nx(self) -> int:
        return len(self.states)

    @property
    def nu(self) -> int:
        return len(self.actions)

    @property
    def n_p(self) -> int:
        return len(self.parameters)

    @property
    def ny(self) -> int:
        return len(self.measurement)

    # ------------------------------------------------------------------
    # Structural validation -- no equation parsing here, see module
    # docstring. Returns a list of plain-language messages; empty list
    # means "structurally sound enough to attempt equation parsing next."
    # Never raises -- a config, including one loaded from a hand-edited
    # or corrupted JSON file, is always safe to call this on.
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []

        def is_valid_identifier(n) -> bool:
            return isinstance(n, str) and bool(IDENTIFIER_RE.match(n))

        if not self.name or not str(self.name).strip():
            errors.append("The model needs a name.")

        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            errors.append("The time step (dt) must be a positive number.")

        if not isinstance(self.n_sessions, int) or self.n_sessions < 1:
            errors.append("The number of steps to simulate must be at least 1.")
        elif self.n_sessions > MAX_N_SESSIONS:
            errors.append(
                f"The number of steps to simulate ({self.n_sessions}) is "
                f"above the maximum of {MAX_N_SESSIONS}. Large horizons can "
                f"make a single run take a very long time -- lower this "
                f"value.")

        if not self.states:
            errors.append("Define at least one state (a quantity your "
                           "system tracks over time).")
        if not self.actions:
            errors.append("Define at least one action (something you can "
                           "choose or control at each step).")

        # Name uniqueness across ALL categories -- equations can't
        # otherwise tell which one a symbol refers to.
        seen: dict[str, str] = {}
        for category, names in (
                ("state", self.state_names()),
                ("action", self.action_names()),
                ("unknown parameter", self.parameter_names()),
                ("constant", self.constant_names())):
            for n in names:
                if not is_valid_identifier(n):
                    errors.append(
                        f"'{n}' is not a valid name for a {category} -- use "
                        f"only letters, numbers, and underscores, and don't "
                        f"start with a number.")
                    continue
                if n in seen and seen[n] != category:
                    errors.append(
                        f"The name '{n}' is used for both a {seen[n]} and a "
                        f"{category} -- every name must be unique across "
                        f"states, actions, unknown parameters, and constants.")
                elif n in seen:
                    errors.append(
                        f"The name '{n}' is used more than once as a "
                        f"{category} -- names must be unique.")
                else:
                    seen[n] = category

        reserved = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh",
                    "abs", "min", "max", "pi", "e"}
        for n in seen:
            if n in reserved:
                errors.append(
                    f"'{n}' is a reserved function/constant name and can't "
                    f"be used as a state/action/parameter/constant name.")

        # Actions need real, ordered bounds -- required for u_lims.
        for a in self.actions:
            if not (isinstance(a.min, (int, float)) and isinstance(a.max, (int, float))):
                errors.append(f"Action '{a.name}' needs numeric minimum and "
                               f"maximum values.")
            elif a.min >= a.max:
                errors.append(f"Action '{a.name}': the minimum ({a.min}) "
                               f"must be less than the maximum ({a.max}).")

        for s in self.states:
            if s.min is not None and s.max is not None and s.min >= s.max:
                errors.append(f"State '{s.name}': the minimum ({s.min}) "
                               f"must be less than the maximum ({s.max}).")
            if not isinstance(s.process_noise, (int, float)) or s.process_noise < 0:
                errors.append(f"State '{s.name}': the process noise scale "
                               f"can't be negative.")

        for p in self.parameters:
            if not isinstance(p.prior_variance, (int, float)) or p.prior_variance <= 0:
                errors.append(f"Unknown parameter '{p.name}': the initial "
                               f"uncertainty must be a positive number.")
            if not isinstance(p.process_noise, (int, float)) or p.process_noise < 0:
                errors.append(f"Unknown parameter '{p.name}': the drift/"
                               f"process noise scale can't be negative.")

        # Dynamics: exactly one equation per state, no extras, no gaps.
        state_names = set(self.state_names())
        dyn_keys = set(self.dynamics.keys())
        missing = state_names - dyn_keys
        extra = dyn_keys - state_names
        for m in sorted(missing):
            errors.append(f"State '{m}' has no equation for how it changes "
                           f"over time.")
        for e in sorted(extra):
            errors.append(f"There's an equation for '{e}', which isn't a "
                           f"defined state.")
        for s_name, expr in self.dynamics.items():
            if not expr or not str(expr).strip():
                errors.append(f"The equation for '{s_name}' is empty.")

        if not self.measurement:
            errors.append("Define at least one thing that can be observed/"
                           "measured about your system.")
        meas_names = [m.name for m in self.measurement]
        if len(meas_names) != len(set(meas_names)):
            errors.append("Two or more measured outputs share the same name "
                           "-- output names must be unique.")
        for m in self.measurement:
            if not m.expression or not str(m.expression).strip():
                errors.append(f"Measured output '{m.name}' has no expression.")
            if not isinstance(m.noise_scale, (int, float)) or m.noise_scale < 0:
                errors.append(f"Measured output '{m.name}': the measurement "
                               f"noise scale can't be negative.")

        if not self.cost.running or not str(self.cost.running).strip():
            errors.append("The running cost expression is empty.")
        if self.cost.terminal is None or not str(self.cost.terminal).strip():
            errors.append("The terminal cost expression is empty (use '0' "
                           "if there's no separate terminal cost).")

        if self.control_method not in CONTROL_METHODS:
            errors.append(
                f"Control method must be one of {CONTROL_METHODS}, got "
                f"'{self.control_method}'.")
        elif (self.control_method == "ilqr" and isinstance(self.n_sessions, int)
                and self.n_sessions < 2):
            errors.append(
                "iLQR control method needs at least 2 steps to simulate "
                "(one state to start from, at least one control step).")

        constraint_names_seen: set[str] = set()
        for c in self.constraints:
            if not c.name or not str(c.name).strip():
                errors.append("Every constraint needs a name.")
            elif c.name in constraint_names_seen:
                errors.append(f"The constraint name '{c.name}' is used more "
                               f"than once -- constraint names must be unique.")
            else:
                constraint_names_seen.add(c.name)
            if c.kind not in CONSTRAINT_KINDS:
                errors.append(
                    f"Constraint '{c.name}': kind must be one of "
                    f"{CONSTRAINT_KINDS}, got '{c.kind}'.")
            if not c.expression or not str(c.expression).strip():
                errors.append(f"Constraint '{c.name}': the expression is empty.")
            if c.kind == "state_only":
                if not isinstance(c.alpha, (int, float)) or c.alpha <= 0:
                    errors.append(
                        f"Constraint '{c.name}': alpha must be a positive "
                        f"number for a state-only constraint (it controls "
                        f"how quickly the constraint's own reduction term "
                        f"decays -- see extensions/constraints/"
                        f"relative_degree_reduction.py).")
        if (self.constraints and any(c.enabled for c in self.constraints)
                and self.solver.u_lim_method != 1):
            errors.append(
                "Constraints require the 'Box-QP' bound enforcement "
                "method (Advanced settings) -- the tanh-squash method "
                "doesn't compose sensibly with state-dependent control "
                "bounds.")

        s = self.solver
        if s.u_lim_method not in (1, 2):
            errors.append("Advanced setting 'bound enforcement method' must "
                           "be 1 or 2.")
        if s.max_du_iterations < 1 or s.max_du_iterations > MAX_DU_ITERATIONS:
            errors.append(
                f"Advanced setting 'planning iterations per step' must be "
                f"between 1 and {MAX_DU_ITERATIONS}.")
        if (s.first_run_max_du_iterations < 1
                or s.first_run_max_du_iterations > MAX_FIRST_RUN_DU_ITERATIONS):
            errors.append(
                f"Advanced setting 'planning iterations for the first step' "
                f"must be between 1 and {MAX_FIRST_RUN_DU_ITERATIONS}.")

        return errors

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)  # don't mutate caller's dict
        states = [StateSpec(**s) for s in d.get("states", [])]
        actions = [ActionSpec(**a) for a in d.get("actions", [])]
        parameters = [ParameterSpec(**p) for p in d.get("parameters", [])]
        constants = [ConstantSpec(**c) for c in d.get("constants", [])]
        measurement = [MeasurementSpec(**m) for m in d.get("measurement", [])]
        cost = CostSpec(**d.get("cost", {}))
        constraints = [ConstraintSpec(**c) for c in d.get("constraints", [])]
        solver = SolverSettings(**d.get("solver", {}))
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            dt=d.get("dt", 1.0),
            n_sessions=d.get("n_sessions", 15),
            control_method=d.get("control_method", "ilqg"),
            states=states,
            actions=actions,
            parameters=parameters,
            constants=constants,
            dynamics=dict(d.get("dynamics", {})),
            measurement=measurement,
            cost=cost,
            constraints=constraints,
            solver=solver,
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )
