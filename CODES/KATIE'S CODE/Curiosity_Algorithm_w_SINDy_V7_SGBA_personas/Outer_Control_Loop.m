%% =========================================================================
%  Outer_Control_Loop.m
%  Dual iLQG rehabilitation therapy planner for traumatic SCI.
%
%  Author : Katie Campbell, UNB ECE 
% =========================================================================

clear; clc; close all;
clear global SINDY_MODEL;
rng(42);

%% ── 0a. Persona selection ────────────────────────────────────────────────
%  Choose a persona to modulate therapy recommendations for this patient.
%  Options: 'high_motivator' | 'frail_elder' | 'young_athlete' |
%           'low_support'    | 'pain_sensitive' | 'default'
%
%  The persona affects:
%    (a) Patient attribute priors used when simulating / re-generating data
%    (b) Therapy cost weights (intensity burden, frequency aversion, goals)
%    (c) The functional ceiling (comorbidity penalty on SWAT)

PERSONA_NAME = 'high_motivator';   % ← change this to switch personas
persona      = define_persona(PERSONA_NAME);

fprintf('─────────────────────────────────────────\n');
fprintf('  Persona : %s\n', persona.name);
fprintf('  %s\n', persona.description);
fprintf('  Motivation: %.2f  |  Fatigue tol: %.2f\n', ...
        persona.motivation, persona.fatigue_tol);
fprintf('  Pain sensitivity: %.2f  |  Comorbidity: %.2f\n', ...
        persona.pain_sensitivity, persona.comorbidity);
fprintf('─────────────────────────────────────────\n\n');

%% ── 0. Load SINDy model ──────────────────────────────────────────────────

LOAD_SINDY   = true;
sindy_path   = 'results/sindy_model.mat';

if LOAD_SINDY && exist(sindy_path, 'file')
    S = load(sindy_path, 'Xi', 'lib_labels', 'cfg', 'dataset');
    Xi_identified = S.Xi;
    lib_cfg       = S.cfg;
    fprintf('Loaded SINDy model from %s\n', sindy_path);
    fprintf('  Non-zero coefficients: %d / %d\n', nnz(Xi_identified), numel(Xi_identified));
else
    warning('SINDy model not found. Using placeholder Xi.');
    lib_cfg = struct('poly_order', 2, 'include_cross', true, 'include_trig', false);
    nx_dyn  = 6;
    nu_dyn  = who_therapies().n;                       % 13 canonical therapy controls
    nvars   = nx_dyn + nu_dyn;                          % variables in the library
    n_terms = 1 + nvars + nvars + nvars*(nvars-1)/2;    % const + linear + squares + cross
    Xi_identified = zeros(n_terms, nx_dyn);
    Xi_identified(1 + nx_dyn + 1, 1) = 0.10;            % first therapy (a1) -> SCIM
    Xi_identified(1 + nx_dyn + 1, 2) = 0.08;            % first therapy (a1) -> BBS
end

Xi_identified(:, 3:6) = 0;

%% ── 1. Patient setup ─────────────────────────────────────────────────────

x0_true = [0.25; 0.20; 0.75; 0.45; 0.30; 0.70];

swat_ceiling = 0.4*x0_true(3) + 0.2*(1-x0_true(4)) + 0.2*x0_true(6) + 0.1*(1-x0_true(5));
% Apply persona comorbidity penalty to the functional ceiling
swat_ceiling = swat_ceiling * (1 - 0.5 * persona.comorbidity);
swat_ceiling = min(max(swat_ceiling, 0.1), 1.0);

fprintf('\nPatient profile:\n');
fprintf('  AIS: C  |  Age norm: %.2f  |  Caregiver: %.2f\n', x0_true(4), x0_true(6));
fprintf('  Initial SCIM: %.2f  |  Initial BBS: %.2f\n', x0_true(1), x0_true(2));
fprintf('  Recovery ceiling (SWAT, persona-adjusted): %.2f\n\n', swat_ceiling);

%% ── 2. Identify uncertain SINDy parameters ───────────────────────────────

% FIX 1: Use np_max=2 (was 4).  np=2 captures the dominant action
% coefficients while keeping the augmented state tractable.
np_max = 2;

[term_idx, state_idx] = find(Xi_identified(:, 1:2) ~= 0);
n_uncertain = min(np_max, length(term_idx));
xi_idx = [term_idx(1:n_uncertain), state_idx(1:n_uncertain)];
np = n_uncertain;

fprintf('Augmented parameters (uncertain SINDy coefficients): %d\n', np);

%% ── 3. System dimensions ─────────────────────────────────────────────────

% Number of therapy control signals — derive from the model/data, never hardcode.
if exist('S', 'var') && isfield(S, 'dataset') && isfield(S.dataset, 'n_actions')
    nu = S.dataset.n_actions;        % e.g. 13 WHO therapies
else
    nu = who_therapies().n;          % canonical count (13)
end
nx = 6; ny = 2; nv = ny; nw = nx + np;
T  = 12; dt = 1; N = T;

horizon_mode   = 2;
horizon_length = 6;
if horizon_mode == 1; horizon_length = N; end

%% ── 4. Control limits ────────────────────────────────────────────────────

u_lims = repmat([0, 1], nu, 1);      % [nu x 2] — every therapy dosed in [0,1]
u_lim_method = 2;

%% ── 5. Store SINDy model in global ──────────────────────────────────────

global SINDY_MODEL;
SINDY_MODEL.Xi           = Xi_identified;
SINDY_MODEL.lib_cfg      = lib_cfg;
SINDY_MODEL.swat_ceiling = swat_ceiling;
SINDY_MODEL.xi_idx       = xi_idx;
SINDY_MODEL.F_dyn        = 1e-3;
SINDY_MODEL.F_param      = 1e-4;
SINDY_MODEL.persona      = persona;   % persona weights available to l_cost

c_vec = [swat_ceiling; 1e-3; 1e-4];

%% ── 6. Initial estimates and covariances ─────────────────────────────────

xi_true = zeros(np, 1);
for k = 1:np
    xi_true(k) = Xi_identified(xi_idx(k,1), xi_idx(k,2));
end

xa_true = [x0_true; xi_true];

cov_X = 0.01;
% FIX 2: Larger initial parameter uncertainty → genuine learning signal
cov_P = diag((0.80 * abs(xi_true) + 0.05).^2);   % was (0.5*|xi|+0.01)^2

x_hat = x0_true + sqrt(cov_X) * randn(nx, 1);
x_hat = max(0, min(1, x_hat));
p_hat = xi_true + chol(cov_P, 'lower') * randn(np, 1);
p_hat = max(0.01, p_hat);   % keep positive (coefficients are positive)

cov_xa_filter = blkdiag(cov_X * eye(nx), cov_P);
cov_xa_iLQG   = cov_xa_filter;

%% ── 7. Noise covariances ─────────────────────────────────────────────────

R_cov = 1e-3 * eye(nv);
% FIX 3: Separate Q for states and parameters
Q_state = 1e-3 * eye(nx);
Q_param = 1e-5 * eye(np);   % was 1e-3 for all: caused parameter divergence
Q_cov   = blkdiag(Q_state, Q_param);
sqrtR   = chol(R_cov, 'lower');
sqrtQ   = chol(Q_cov, 'lower');

%% ── 8. iLQG algorithm options ────────────────────────────────────────────

lambda_i   = 1; dlambda_i = 1; reset_lambda = 1;
regType    = 3;

augment_states_in_iLQG   = 1;
augment_states_in_filter = 1;

first_run_seeds          = 3;
u_seed_mean              = 0.5;
u_seed_cov               = 0.25^2;
first_run_max_iterations = 200;
max_du_iterations        = 100;

verbose               = 1;
keep_dyn_noise_filter = 1;
keep_meas_noise_filter = 1;
h_spkf               = sqrt(3);
AdditiveNoise        = 0;

%% ── 9. Initialise trajectories ───────────────────────────────────────────

xa_true_all = zeros(nx + np, N + 1);
x_hat_all   = zeros(nx,      N + 1);
p_hat_all   = zeros(np,      N + 1);
u_all       = zeros(nu,      N);
cov_P_all   = zeros(np, np, N + 1);

xa_true_all(:, 1) = xa_true;
x_hat_all(:, 1)   = x_hat;
p_hat_all(:, 1)   = p_hat;
cov_P_all(:,:,1)  = cov_P;

lambda  = lambda_i;
dlambda = dlambda_i;

l_gain = zeros(nu, horizon_length);
L_gain = zeros(nu, nx + np, horizon_length);
u_bar  = u_seed_mean * ones(nu, horizon_length);
Tracking_Trajectory = [];

%% ── 10. Main session loop ─────────────────────────────────────────────────

fprintf('\nRunning dual iLQG over %d sessions...\n\n', N);

for s = 1:N

    T_rem_s = min(horizon_length, N - s + 1);
    xa_hat_s = x_hat_all(:, s);    % nx rows only — iLQG appends p_hat internally
    p_hat_s  = p_hat_all(:, s);

    %% iLQG planning
    if s == 1
        best_cost = Inf; Pw = []; Pv = [];
        for seed = 1:first_run_seeds
            l_s = u_seed_mean + sqrt(u_seed_cov) * randn(nu, T_rem_s);
            L_s = zeros(nu, nx + np, T_rem_s);
            try
                [~, unew_s, l_s, L_s, ~, ~, cost_s, Pw_s, Pv_s, ~] = ...
                    iLQG_function(T_rem_s, 1, xa_hat_s, l_s, L_s, l_s, ...
                                  lambda, dlambda, c_vec, p_hat_s, cov_xa_iLQG, ...
                                  augment_states_in_iLQG, regType, u_lims, ...
                                  ny, nv, nw, first_run_max_iterations, verbose, ...
                                  0, 0, Tracking_Trajectory, u_lim_method);
                if sum(cost_s) < best_cost
                    best_cost = sum(cost_s); unew = unew_s;
                    l_gain = l_s; L_gain = L_s; Pw = Pw_s; Pv = Pv_s;
                end
            catch
            end
        end
        if isempty(Pw)
            [~, unew, l_gain, L_gain, ~, ~, ~, Pw, Pv, ~] = ...
                iLQG_function(T_rem_s, 1, xa_hat_s, l_gain, L_gain, u_bar, ...
                              lambda, dlambda, c_vec, p_hat_s, cov_xa_iLQG, ...
                              augment_states_in_iLQG, regType, u_lims, ...
                              ny, nv, nw, first_run_max_iterations, verbose, ...
                              0, 0, Tracking_Trajectory, u_lim_method);
        end
    else
        [~, unew, l_gain, L_gain, ~, ~, ~, Pw, Pv, ~] = ...
            iLQG_function(T_rem_s, 1, xa_hat_s, l_gain, L_gain, u_bar, ...
                          lambda, dlambda, c_vec, p_hat_s, cov_xa_iLQG, ...
                          augment_states_in_iLQG, regType, u_lims, ...
                          ny, nv, nw, max_du_iterations, verbose, ...
                          0, 0, Tracking_Trajectory, u_lim_method);
    end

    % ── Squash raw iLQG output to physical [0,1] bounds ──────────────────
    % u_lim_method=2 uses tanh squashing inside the cost, but the returned
    % unew is still in the unconstrained pre-tanh space.  Apply the same
    % squash here so that what we log, simulate, and warm-start with is
    % always a physically valid action.
    u_raw = unew(:, 1);
    if u_lim_method == 2 && ~isempty(u_lims)
        u_phys = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u_raw) + ...
                 (u_lims(:,2) + u_lims(:,1))/2;
    else
        u_phys = max(u_lims(:,1), min(u_lims(:,2), u_raw));
    end
    u_all(:, s) = u_phys;

    if reset_lambda; lambda = lambda_i; dlambda = dlambda_i; end

    if s == N; break; end   % no more states to update after final session

    %% Simulate true system
    % Use the physically squashed action (u_all already clamped above)
    [y_obs, xa_sim] = simulate_system(1, xa_true_all(:,s), u_all(:,s), ...
                                       c_vec, sqrtR, sqrtQ, ...
                                       augment_states_in_filter, 0, ...
                                       u_lims, u_lim_method);
    xa_true_all(:, s+1) = xa_sim(:, end);

    %% SPKF filter update
    xa_hat_filt = [x_hat_all(:,s); p_hat_all(:,s)];

    % FIX 4: Floor Q on both blocks before passing to SPKF
    Pw_f = Pw(:,:,1);
    Pw_f(1:nx,    1:nx)    = max(Pw_f(1:nx,1:nx),    1e-4*eye(nx));
    Pw_f(nx+1:end,nx+1:end)= max(Pw_f(nx+1:end,nx+1:end), 1e-5*eye(np));

    Dyn_F  = @(dt_f, xk, wk) DiscreteStateDynamics(dt_f, xk, u_all(:,s), ...
                 c_vec, wk, 0, augment_states_in_filter, 1, 0, ...
                 u_lims, u_lim_method);
    Meas_F = @(xk, vk) Measurement(1, xk, u_all(:,s), c_vec, vk, 0, ...
                 augment_states_in_filter, 1);

    [xa_filt_all, P_filt] = SPKF_function(1, Dyn_F, Meas_F, y_obs(end,:)', ...
                                        xa_hat_filt, zeros(nx+np,1), ...
                                        zeros(ny,1), cov_xa_filter, ...
                                        Pw_f * keep_dyn_noise_filter, ...
                                        Pv(:,:,1) * keep_meas_noise_filter, ...
                                        h_spkf, AdditiveNoise);

    % SPKF_function returns [iterations x n_states] — extract last row as column
    xa_filt = xa_filt_all(end, :)';

    % FIX 4: Floor updated covariance
    P_filt = 0.5*(P_filt + P_filt');
    P_filt(1:nx,    1:nx)    = max(P_filt(1:nx,1:nx),    1e-5*eye(nx));
    P_filt(nx+1:end,nx+1:end)= max(P_filt(nx+1:end,nx+1:end), 1e-6*eye(np));

    x_hat_all(:, s+1) = xa_filt(1:nx);
    p_hat_all(:, s+1) = max(0.01, xa_filt(nx+1:end));
    cov_xa_filter      = P_filt;
    cov_xa_iLQG        = P_filt;
    cov_P_all(:,:,s+1) = P_filt(nx+1:end, nx+1:end);

    % Warm start — shift plan left by one step, hold last action.
    % Apply the same squash so u_bar stays in physical space and doesn't
    % feed back a degenerate pre-tanh value that locks the iLQG.
    if size(unew, 2) > 1
        u_bar_raw = [unew(:,2:end), unew(:,end)];
        if u_lim_method == 2 && ~isempty(u_lims)
            u_bar = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u_bar_raw) + ...
                    (u_lims(:,2) + u_lims(:,1))/2;
        else
            u_bar = max(u_lims(:,1), min(u_lims(:,2), u_bar_raw));
        end
    end

    if verbose
        [top_dose, top_j] = max(u_all(:,s));
        fprintf('Session %2d | mean dose=%.2f | top: U%d=%.2f | SCIM=%.3f BBS=%.3f | param_err=%.4f\n', ...
            s, mean(u_all(:,s)), top_j, top_dose, ...
            xa_true_all(1,s+1), xa_true_all(2,s+1), ...
            norm(p_hat_all(:,s+1) - xi_true));
    end
end

%% ── 11. Results summary ──────────────────────────────────────────────────

init_score  = (xa_true_all(1,1) + xa_true_all(2,1)) / 2;
final_score = (xa_true_all(1,N) + xa_true_all(2,N)) / 2;
ceiling_gap = max(swat_ceiling - init_score, 0.01);
MFG = (final_score - init_score) / ceiling_gap;

fprintf('\n══════════════════════════════════\n');
fprintf('  MFG   : %.4f\n', MFG);
fprintf('  Init  : SCIM=%.3f  BBS=%.3f\n', xa_true_all(1,1), xa_true_all(2,1));
fprintf('  Final : SCIM=%.3f  BBS=%.3f\n', xa_true_all(1,N), xa_true_all(2,N));
fprintf('  Param error session 1 → %d : %.4f → %.4f\n', N, ...
        norm(p_hat_all(:,1)-xi_true), norm(p_hat_all(:,N)-xi_true));
fprintf('══════════════════════════════════\n');

%% ── 12. Plots ────────────────────────────────────────────────────────────

figure('Name','Dual iLQG Results','Position',[100,100,1200,400]);

subplot(1,3,1);
plot(1:N+1, xa_true_all(1,:), 'b-o', 'DisplayName','SCIM true'); hold on;
plot(1:N+1, xa_true_all(2,:), 'r-s', 'DisplayName','BBS true');
plot(1:N+1, x_hat_all(1,:),   'b--', 'DisplayName','SCIM est');
plot(1:N+1, x_hat_all(2,:),   'r--', 'DisplayName','BBS est');
yline(swat_ceiling, 'k:', 'SWAT ceiling');
xlabel('Session'); ylabel('Score [0,1]'); title('State trajectories');
legend('Location','southeast'); grid on;

subplot(1,3,2);
TH_codes = who_therapies().codes;
mean_dose = mean(u_all, 2);                      % [nu x 1] mean recommended dose
bar(mean_dose, 'FaceColor', [0.06 0.43 0.34]);
set(gca, 'XTick', 1:numel(TH_codes), 'XTickLabel', TH_codes);
xtickangle(60);
xlabel('Therapy'); ylabel('Mean dose [0,1]');
title('Recommended therapy mix'); ylim([0,1]); grid on;

subplot(1,3,3);
for k = 1:np
    plot(1:N+1, p_hat_all(k,:), '-o', 'DisplayName', sprintf('\\xi_%d est',k)); hold on;
    yline(xi_true(k), '--', sprintf('\\xi_%d true',k));
end
trace_P = squeeze(sum(sum(cov_P_all .* repmat(eye(np),[1,1,N+1]), 1), 2))';
yyaxis right;
plot(1:N+1, trace_P, 'k:', 'DisplayName','trace(\Sigma_p)');
xlabel('Session'); title('Parameter convergence'); legend; grid on;

sgtitle('Dual iLQG: Curiosity-Driven SCI Rehabilitation Planner', 'FontWeight','bold');
