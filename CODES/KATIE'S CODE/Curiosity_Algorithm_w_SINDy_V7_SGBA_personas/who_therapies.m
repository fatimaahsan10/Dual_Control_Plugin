function T = who_therapies()
%WHO_THERAPIES  Central definition of the 13 control signals (therapies).
%
%  Replaces the original 3 abstract controls (Intensity / Modality /
%  Frequency) with 13 concrete rehabilitation interventions drawn from the
%  WHO "Package of Interventions for Rehabilitation" (Spinal Cord Injury
%  module). Each control signal u_j in [0,1] is the *dose* of that therapy
%  prescribed in a given session (0 = not used, 1 = maximum dose).
%
%  This single source of truth is consumed by:
%    - generate_synthetic_dataset.m  (ground-truth dynamics)
%    - define_persona.m              (per-therapy efficacy profiles)
%    - run_persona_comparison.m      (reporting / plotting)
%    - print_identified_equations.m  (human-readable equation labels)
%
%  Functional outcome model
%  ────────────────────────
%  Each therapy contributes to the two dynamic functional states:
%      row 1 of G : SCIM (independence / motor function) gain per unit dose
%      row 2 of G : BBS  (balance) gain per unit dose
%  The values below encode the primary clinical target of each therapy
%  (e.g. balance training loads BBS heavily; hand/arm training loads SCIM
%  and almost nothing on balance). They are population-average "true"
%  gains; persona-specific efficacy (define_persona) then scales them so
%  that the optimiser recommends a *different* therapy mix per persona.
%
%  load : physical demand of each therapy in [0,1]. High-load therapies are
%         attenuated for fatigue-limited / pain-sensitive personas and incur
%         marginally higher resource cost.
%
%  Author : Katie Campbell, UNB ECE
%  Date   : June 2026

%            U1    U2    U3    U4    U5    U6    U7    U8    U9    U10   U11   U12   U13
T.codes  = {'U1','U2','U3','U4','U5','U6','U7','U8','U9','U10','U11','U12','U13'};

T.names = { ...
    'Range of motion exercises', ...                                 % U1
    'Positioning for contracture prevention (incl. weight bearing)', ... % U2
    'Muscle-strengthening exercises', ...                            % U3
    'Neuromuscular / functional electrical stimulation (NMES/FES)', ... % U4
    'Antispastic pattern positioning', ...                           % U5
    'Stretching', ...                                                % U6
    'Balance training', ...                                          % U7
    'Gait training', ...                                             % U8
    'Assistive products for mobility (provision + training)', ...    % U9
    'Mobility training (incl. wheelchair skills)', ...               % U10
    'Functional positioning', ...                                    % U11
    'Functional training for hand and arm use', ...                  % U12
    'Fitness training' };                                            % U13

% Assessment grouping (WHO model) — for grouped reporting only.
T.assessment = { ...
    'Joint mobility','Joint mobility','Joint mobility','Joint mobility', ...
    'Muscle tone','Muscle tone', ...
    'Balance', ...
    'Gait & walking','Gait & walking', ...
    'Mobility','Mobility', ...
    'Hand & arm use', ...
    'Exercise capacity' };

%% ── Base functional gains  G = [SCIM; BBS]  (2 x 13) ─────────────────────
%        U1    U2    U3    U4    U5    U6    U7    U8    U9    U10   U11   U12   U13
G_scim = [0.15, 0.10, 0.30, 0.28, 0.08, 0.12, 0.10, 0.30, 0.22, 0.25, 0.15, 0.26, 0.18];
G_bbs  = [0.08, 0.05, 0.15, 0.08, 0.06, 0.08, 0.35, 0.25, 0.06, 0.08, 0.10, 0.02, 0.15];

T.G = [G_scim; G_bbs];

%% ── Physical load (demand) per therapy in [0,1] ─────────────────────────
%   Strengthening, NMES, gait, balance and fitness are the most demanding.
%        U1    U2    U3    U4    U5    U6    U7    U8    U9    U10   U11   U12   U13
T.load = [0.30, 0.15, 0.85, 0.70, 0.10, 0.25, 0.55, 0.80, 0.20, 0.40, 0.15, 0.45, 0.75];

T.n = numel(T.codes);   % = 13

end
