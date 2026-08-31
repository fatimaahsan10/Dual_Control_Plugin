function dataset = generate_synthetic_dataset(cfg)
%GENERATE_SYNTHETIC_DATASET  Generate a clinically grounded synthetic SCI
%  rehabilitation dataset for SINDy pipeline development and validation.
%
%  13-THERAPY EXTENSION (June 2026)
%  --------------------------------
%  The 3 abstract controls (Intensity/Modality/Frequency) are replaced by 13
%  concrete WHO rehabilitation therapies (see who_therapies.m). Each control
%  u_j in [0,1] is the dose of therapy U_j prescribed in a session. Each
%  therapy contributes to the two dynamic functional states via a base gain
%  matrix G (2 x 13). Persona-specific per-therapy efficacy (therapy_efficacy)
%  scales those gains so that the optimal therapy MIX differs by persona -
%  this is what lets the curiosity-driven controller recommend a different,
%  appropriate plan for each persona.
%
%  PERSONA MODULATION
%  ------------------
%  cfg.persona (from define_persona) modulates:
%    - Patient attribute sampling (AIS priors, age, DPI, caregiver)
%    - Per-therapy effective dose: motivation (global gain), therapy_efficacy
%      (per-therapy response), pain_sensitivity & fatigue_tol (attenuate
%      high physical-load therapies), comorbidity (ceiling penalty)
%
%  State variables (n_states = 6):
%    x1 : SCIM III composite score   [0,1 normalised]  (independence/motor)
%    x2 : Berg Balance Scale (BBS)   [0,1 normalised]  (balance)
%    x3 : AIS classification         {0.25,0.5,0.75,1.0} (A/B/C/D)  [fixed]
%    x4 : Age (normalised)           [0,1]                          [fixed]
%    x5 : Days post-injury (norm.)   [0,1]                          [fixed]
%    x6 : Caregiver support score    [0,1]                          [fixed]
%
%  Action variables (n_actions = 13):  U1..U13 (who_therapies)
%
%  True governing dynamics (persona-modulated), j = 1..13:
%    u_eff_j = u_j * eff_j * motivation * load_atten_j
%    dx1/dt  = (-0.05*x1 + 0.30*x1*x3 + (G_scim . u_eff)*(1+0.30*x3+0.15*x6)) * sat1
%    dx2/dt  = (-0.04*x2 + 0.22*x2*x3 + (G_bbs  . u_eff)*(1+0.15*x6))         * sat2
%    sat_i   = max(0, ceiling_adjusted - x_i)

n_p = cfg.n_patients;
n_s = cfg.n_sessions;
noise_std = cfg.noise_std;

%% -- Persona parameters ---------------------------------------------------

if isfield(cfg, 'persona') && isstruct(cfg.persona)
    prs = cfg.persona;
else
    prs = define_persona('default');
end

%% -- WHO therapy definition (13 control signals) --------------------------

TH       = who_therapies();
n_actions = TH.n;              % 13
G_scim   = TH.G(1, :);         % 1 x 13 base SCIM gains
G_bbs    = TH.G(2, :);         % 1 x 13 base BBS  gains
load     = TH.load;            % 1 x 13 physical-load profile
eff      = prs.therapy_efficacy(:)';   % 1 x 13 persona efficacy

%% -- Patient attribute sampling (persona-modulated) -----------------------

ais_vals  = [0.25, 0.50, 0.75, 1.0];
ais_idx   = randsample(1:4, n_p, true, prs.ais_probs);
ais       = ais_vals(ais_idx)';

age_lo  = prs.age_range(1);
age_hi  = prs.age_range(2);
age_raw = max(age_lo, min(age_hi, age_lo + (age_hi - age_lo) * betarnd(2, 3, n_p, 1)));
age     = (age_raw - age_lo) / max(age_hi - age_lo, 1);

dpi_lo  = prs.dpi_range(1);
dpi_hi  = prs.dpi_range(2);
dpi_raw = dpi_lo + (dpi_hi - dpi_lo) * rand(n_p, 1);
dpi     = (dpi_raw - dpi_lo) / max(dpi_hi - dpi_lo, 1);

cg      = betarnd(prs.caregiver_alpha, prs.caregiver_beta, n_p, 1);

scim0     = max(0.01, ais * 0.5 + 0.1*randn(n_p,1));
scim0     = min(scim0, 1.0);

bbs0      = max(0.01, ais * 0.45 + 0.1*scim0 + 0.08*randn(n_p,1));
bbs0      = min(bbs0, 1.0);

% Comorbidity reduces the achievable functional ceiling
swat_ceiling = 0.4*ais + 0.2*(1-age) + 0.2*cg + 0.1*(1-dpi) + 0.1*rand(n_p,1);
swat_ceiling = swat_ceiling * (1 - 0.5 * prs.comorbidity);   % persona ceiling penalty
swat_ceiling = min(max(swat_ceiling, 0.1), 1.0);

%% -- Simulate trajectories ------------------------------------------------

n_states   = 6;
n_outcomes = 1;

X = zeros(n_states,  n_s, n_p);
A = zeros(n_actions, n_s, n_p);
Y = zeros(n_outcomes,n_s, n_p);

for p = 1:n_p
    x3 = ais(p); x4 = age(p); x5 = dpi(p); x6 = cg(p);
    ceil_p = swat_ceiling(p);

    x = [scim0(p); bbs0(p); x3; x4; x5; x6];
    X(:, 1, p) = x + noise_std * randn(n_states, 1);

    for s = 1:n_s - 1
        % SPARSE per-session therapy activation. In real rehabilitation a
        % patient receives a handful of therapies per session, not all 13 at
        % once. Activating a random subset each session (a) matches clinical
        % practice and (b) decorrelates the 13 dose regressors so that each
        % therapy's effect is individually identifiable by SINDy (with all
        % 13 active simultaneously every session the effects are collinear
        % and cannot be separated from one-step finite differences).
        n_active = randi([3, 5]);                 % 3-5 therapies this session
        active   = randperm(n_actions, n_active);
        u = zeros(n_actions, 1);
        u(active) = clip(0.40 + 0.60*rand(n_active, 1), 0, 1);

        % Persona-modulated per-therapy effective dose:
        %   - eff_j        : persona responds better/worse to therapy j
        %   - motivation   : global response scale
        %   - load_atten   : pain & fatigue attenuate HIGH-LOAD therapies
        pain_atten    = 1 - prs.pain_sensitivity * (load(:) .* max(0, u - 0.60));
        fatigue_atten = 1 - (1 - prs.fatigue_tol) * 0.5 * load(:);
        load_atten    = max(0, pain_atten .* fatigue_atten);

        u_eff = u .* eff(:) .* prs.motivation .* load_atten;   % 13 x 1
        u_eff = clip(u_eff, 0, 1.5);

        A(:, s, p) = u;   % store the PRESCRIBED doses (what a clinician sets)

        x1 = x(1); x2 = x(2);

        % Therapy contribution to the per-session functional change. The
        % contribution is (approximately) LINEAR in the doses so that each
        % therapy's coefficient is recoverable by the polynomial SINDy
        % library. Diminishing returns near the recovery ceiling are modelled
        % by a gentle headroom factor head_i = clip(1 - x_i/ceiling, 0, 1)
        % that multiplies the homeostatic decay/state terms strongly and the
        % therapy terms only weakly (so the bare gains stay identifiable).
        head1 = clip(1 - x1 / max(ceil_p, 0.05), 0, 1);
        head2 = clip(1 - x2 / max(ceil_p, 0.05), 0, 1);

        contrib_scim = (G_scim * u_eff) * (1 + 0.30*x3 + 0.15*x6);
        contrib_bbs  = (G_bbs  * u_eff) * (1 + 0.15*x6);

        % Action term scaled by a mild, near-flat headroom (0.55..1.0 over the
        % working range) so it never collapses to zero; state terms keep the
        % stronger saturation.
        soft1 = 0.55 + 0.45*head1;
        soft2 = 0.55 + 0.45*head2;

        dx1 = (-0.05*x1 + 0.30*x1*x3) * head1 + contrib_scim * soft1 * 0.20;
        dx2 = (-0.04*x2 + 0.22*x2*x3) * head2 + contrib_bbs  * soft2 * 0.20;

        x(1) = clip(x(1) + dx1, 0, 1);
        x(2) = clip(x(2) + dx2, 0, 1);

        X(:, s+1, p) = x + noise_std * randn(n_states, 1);
        X(:, s+1, p) = max(0, min(1, X(:, s+1, p)));

        Y(1, s, p) = (dx1 + dx2) / 2 + noise_std * randn();
    end

    A(:, n_s, p) = A(:, n_s-1, p);
end

%% -- Build Xi_true (approximate ground truth, diagnostic) ------------------
% Populate the dominant linear therapy coefficients and the base state terms
% so validate_sindy_model.m and coefficient-recovery diagnostics work. The
% representative linear coefficient for therapy j is G*eff_j*motivation
% (the saturation factor and cross terms add nonlinearity not captured here).

cfg_label = struct('poly_order',2,'include_cross',true,'include_trig',false, ...
                   'n_patients',2,'n_sessions',2,'noise_std',0);
[~, ~, lib_labels_tmp, ~] = build_sindy_library(X(:,:,1:2), A(:,:,1:2), cfg_label);
n_terms = length(lib_labels_tmp);

Xi_true = zeros(n_terms, n_states);

% Strip scale tags for clean matching
clean_labels = regexprep(lib_labels_tmp, '\s*\[s=[\d.]+\]', '');

for ti = 1:n_terms
    lbl = strtrim(clean_labels{ti});

    % Base (action-free) state terms
    switch lbl
        case 'x1';    Xi_true(ti,1) = -0.05;
        case 'x1*x3'; Xi_true(ti,1) =  0.30;
        case 'x2';    Xi_true(ti,2) = -0.04;
        case 'x2*x3'; Xi_true(ti,2) =  0.22;
        case 'x3*x2'; Xi_true(ti,2) =  0.22;
    end

    % Linear therapy terms a1..a13.
    % Representative linear coefficient at a mid working point: the action
    % term enters as G*eff*motivation*soft*0.20 with soft in [0.55,1.0]
    % (~0.8 mid-range), and an interaction lift (1 + 0.30*x3 + 0.15*x6).
    % We report the dominant bare coefficient G*eff*motivation*0.20*0.8.
    tok = regexp(lbl, '^a(\d+)$', 'tokens', 'once');
    if ~isempty(tok)
        j = str2double(tok{1});
        if j >= 1 && j <= n_actions
            base = eff(j) * prs.motivation * 0.20 * 0.80;
            Xi_true(ti,1) = G_scim(j) * base;
            Xi_true(ti,2) = G_bbs(j)  * base;
        end
    end
end
Xi_true(:, 3:6) = 0;   % static states have no dynamics

%% -- Package dataset ------------------------------------------------------

dataset.X             = X;
dataset.A             = A;
dataset.Y             = Y;
dataset.Xi_true       = Xi_true;
dataset.swat_ceiling  = swat_ceiling;
dataset.state_names   = {'SCIM_norm', 'BBS_norm', 'AIS', ...
                          'Age_norm', 'DPI_norm', 'Caregiver'};
dataset.action_names  = TH.codes;             % {'U1',...,'U13'}
dataset.therapy_names = TH.names;             % full descriptions
dataset.outcome_names = {'FunctionalGain'};
dataset.n_states      = n_states;
dataset.n_actions     = n_actions;            % 13
dataset.n_outcomes    = n_outcomes;
dataset.n_patients    = n_p;
dataset.n_sessions    = n_s;
dataset.persona       = prs;

end

%% -- Local helper ---------------------------------------------------------

function v = clip(v, lo, hi)
    v = max(lo, min(hi, v));
end
