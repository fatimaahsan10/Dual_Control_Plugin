%% =========================================================================
%  run_ilqg_cohort.m 
%  Evaluate dual iLQG controller on the 100-patient held-out cohort and
%  compare MFG against the clinical best-practice baseline.
%
% =========================================================================

clear; clc; close all;
clear global SINDY_MODEL;
rng(42);

%% ── 0. Configuration ─────────────────────────────────────────────────────

out_dir = fullfile('results', 'paper');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

horizon_length        = 6;
max_du_iterations     = 100;
first_run_max_iter    = 200;
first_run_seeds       = 3;
u_seed_mean           = 0.5;
u_seed_cov            = 0.25^2;
regType               = 3;
lambda_i              = 1;
dlambda_i             = 1;
reset_lambda          = 1;
u_lim_method          = 2;
augment_states_in_iLQG   = 1;
augment_states_in_filter = 1;
keep_dyn_noise_filter = 1;
keep_meas_noise_filter = 1;
h_spkf                = sqrt(3);
AdditiveNoise         = 0;
np_max                = 2;
verbose_ilqg          = 0;

%% ── 1. Load SINDy model ──────────────────────────────────────────────────

sindy_path = fullfile('results', 'sindy_model.mat');
if ~exist(sindy_path, 'file')
    error('sindy_model.mat not found. Run MAIN_sindy_pipeline.m first.');
end
S = load(sindy_path, 'Xi', 'lib_labels', 'cfg');
Xi_identified         = S.Xi;
Xi_identified(:, 3:6) = 0;
lib_cfg               = S.cfg;

fprintf('Loaded SINDy model: %d non-zero terms\n', nnz(Xi_identified));

%% ── 2. Generate evaluation cohort ───────────────────────────────────────

fprintf('Generating 500-patient dataset...\n');
cfg_data               = lib_cfg;
cfg_data.n_patients    = 500;
cfg_data.n_sessions    = 12;
cfg_data.noise_std     = 0.05;
cfg_data.verbose       = false;
full_dataset           = generate_synthetic_dataset(cfg_data);

n_eval   = 100;
eval_idx = (500 - n_eval + 1):500;

X_eval    = full_dataset.X(:, :, eval_idx);
swat_eval = full_dataset.swat_ceiling(eval_idx);

fprintf('Evaluation cohort: %d patients\n', n_eval);

%% ── 3. Clinical baseline ─────────────────────────────────────────────────
% FIX 5: Use same true dynamics path as iLQG patients for fair comparison.

ais_protocol = [0.25, 0.40, 0.50, 0.60;
                0.50, 0.55, 0.50, 0.70;
                0.75, 0.70, 0.50, 0.80;
                1.00, 0.80, 0.50, 0.85];

gain_baseline = zeros(n_eval, 1);

for p = 1:n_eval
    x0_p    = X_eval(:, 1, p);
    ceil_p  = swat_eval(p);
    ais_val = x0_p(3);
    [~, ais_row] = min(abs(ais_protocol(:,1) - ais_val));
    a_bl = ais_protocol(ais_row, 2:4)';

    % Use the SINDY_MODEL global for consistent dynamics
    global SINDY_MODEL;
    SINDY_MODEL.Xi           = Xi_identified;
    SINDY_MODEL.lib_cfg      = lib_cfg;
    SINDY_MODEL.swat_ceiling = ceil_p;
    SINDY_MODEL.xi_idx       = zeros(0, 2);   % no uncertain params for baseline
    SINDY_MODEL.F_dyn        = 1e-3;
    SINDY_MODEL.F_param      = 1e-4;

    c_vec_bl = [ceil_p; 1e-3; 1e-4];
    sqrtR_bl = sqrt(1e-3) * eye(2);
    sqrtQ_bl = sqrt(1e-3) * eye(6);   % nx=6, np=0 → nw=6

    x_bl = x0_p;
    for s = 1:cfg_data.n_sessions - 1
        [~, xa_bl] = simulate_system(1, x_bl, a_bl, c_vec_bl, sqrtR_bl, sqrtQ_bl, ...
                                      0, 0, build_u_lims(3), u_lim_method);
        x_bl = xa_bl(:, end);
        x_bl = max(0, min(1, x_bl));
    end
    init_score  = (X_eval(1,1,p) + X_eval(2,1,p)) / 2;
    final_score = (x_bl(1) + x_bl(2)) / 2;
    ceiling_gap = max(ceil_p - init_score, 0.01);
    gain_baseline(p) = (final_score - init_score) / ceiling_gap;
end

fprintf('Clinical baseline MFG: %.4f ± %.4f\n', mean(gain_baseline), std(gain_baseline));

%% ── 4. Dual iLQG cohort evaluation ──────────────────────────────────────

gain_ilqg         = zeros(n_eval, 1);
param_uncertainty = zeros(n_eval, cfg_data.n_sessions);
param_error       = zeros(n_eval, cfg_data.n_sessions);

fprintf('\nRunning dual iLQG on %d evaluation patients...\n', n_eval);

for p = 1:n_eval

    fprintf('Patient %3d / %d ', p, n_eval);

    global SINDY_MODEL;
    swat_p = swat_eval(p);

    [term_idx, state_idx] = find(Xi_identified(:, 1:2) ~= 0);
    n_uncertain = min(np_max, length(term_idx));
    xi_idx_p    = [term_idx(1:n_uncertain), state_idx(1:n_uncertain)];
    np_p        = n_uncertain;

    SINDY_MODEL.Xi           = Xi_identified;
    SINDY_MODEL.lib_cfg      = lib_cfg;
    SINDY_MODEL.swat_ceiling = swat_p;
    SINDY_MODEL.xi_idx       = xi_idx_p;
    SINDY_MODEL.F_dyn        = 1e-3;
    SINDY_MODEL.F_param      = 1e-4;

    c_vec = [swat_p; 1e-3; 1e-4];

    x0_true = X_eval(:, 1, p);

    xi_true_p = zeros(np_p, 1);
    for k = 1:np_p
        xi_true_p(k) = Xi_identified(xi_idx_p(k,1), xi_idx_p(k,2));
    end
    xa_true = [x0_true; xi_true_p];

    % FIX 3: Larger initial parameter uncertainty (80% + 0.05 floor)
    cov_X   = 0.01;
    cov_P_p = diag((0.80 * abs(xi_true_p) + 0.05).^2);

    x_hat = max(0, min(1, x0_true + sqrt(cov_X) * randn(6, 1)));
    p_hat = xi_true_p + chol(cov_P_p, 'lower') * randn(np_p, 1);
    p_hat = max(0.01, p_hat);   % keep physically plausible (positive coefficients)

    cov_xa = blkdiag(cov_X * eye(6), cov_P_p);

    nx = 6; nu = 3; ny = 2; nv = 2; nw = nx + np_p;

    R_cov  = 1e-3 * eye(nv);
    % FIX 4: Reduced Q for parameters to prevent covariance divergence
    Q_state = 1e-3 * eye(nx);
    Q_param = 1e-5 * eye(np_p);     % <-- was 1e-3, causing divergence
    Q_aug   = blkdiag(Q_state, Q_param);
    sqrtR   = chol(R_cov, 'lower');
    sqrtQ   = chol(Q_aug, 'lower');

    T_rem  = horizon_length * ones(1, cfg_data.n_sessions + 1);
    l_gain = zeros(nu, horizon_length);
    L_gain = zeros(nu, nx + np_p, horizon_length);
    u_bar  = u_seed_mean * ones(nu, horizon_length);
    u_all  = zeros(nu, cfg_data.n_sessions);
    Tracking_Trajectory = [];
    lambda  = lambda_i;
    dlambda = dlambda_i;

    xa_true_traj = zeros(nx + np_p, cfg_data.n_sessions + 1);
    xa_true_traj(:, 1) = xa_true;
    x_hat_traj = zeros(nx, cfg_data.n_sessions + 1);
    p_hat_traj = zeros(np_p, cfg_data.n_sessions + 1);
    x_hat_traj(:, 1) = x_hat;
    p_hat_traj(:, 1) = p_hat;

    param_uncertainty(p, 1) = trace(cov_P_p);
    param_error(p, 1)       = norm(p_hat - xi_true_p);

    %% Session loop
    for s = 1:cfg_data.n_sessions - 1

        xa_hat_s = x_hat_traj(:, s);   % pass ONLY x_hat (nx rows) — not [x;p]
        p_hat_s  = p_hat_traj(:, s);

        % FIX 1: Use build_u_lims(nu) consistently — u_lims_mat() is undefined
        u_lims_s = build_u_lims(nu);

        if s == 1
            best_cost = Inf; Pw = []; Pv = [];
            for seed = 1:first_run_seeds
                l_s = u_seed_mean + sqrt(u_seed_cov) * randn(nu, horizon_length);
                L_s = zeros(nu, nx + np_p, horizon_length);
                try
                    [~, unew_s, l_s, L_s, ~, ~, cost_s, Pw_s, Pv_s, ~] = ...
                        iLQG_function(T_rem(s), 1, xa_hat_s, l_s, L_s, l_s, ...
                                      lambda, dlambda, c_vec, p_hat_s, cov_xa, ...
                                      augment_states_in_iLQG, regType, u_lims_s, ...
                                      ny, nv, nw, first_run_max_iter, verbose_ilqg, ...
                                      0, 0, Tracking_Trajectory, u_lim_method);
                    if sum(cost_s) < best_cost
                        best_cost = sum(cost_s);
                        unew = unew_s; l_gain = l_s; L_gain = L_s;
                        Pw = Pw_s; Pv = Pv_s;
                    end
                catch ME
                    fprintf('  seed %d failed: %s\n', seed, ME.message);
                end
            end
            if isempty(Pw)
                % FIX 1: Fallback also uses build_u_lims (not undefined helpers)
                [~, unew, l_gain, L_gain, ~, ~, ~, Pw, Pv, ~] = ...
                    iLQG_function(T_rem(s), 1, xa_hat_s, l_gain, L_gain, u_bar, ...
                                  lambda, dlambda, c_vec, p_hat_s, cov_xa, ...
                                  augment_states_in_iLQG, regType, u_lims_s, ...
                                  ny, nv, nw, first_run_max_iter, verbose_ilqg, ...
                                  0, 0, Tracking_Trajectory, u_lim_method);
            end
        else
            [~, unew, l_gain, L_gain, ~, ~, ~, Pw, Pv, ~] = ...
                iLQG_function(T_rem(s), 1, xa_hat_s, l_gain, L_gain, u_bar, ...
                              lambda, dlambda, c_vec, p_hat_s, cov_xa, ...
                              augment_states_in_iLQG, regType, u_lims_s, ...
                              ny, nv, nw, max_du_iterations, verbose_ilqg, ...
                              0, 0, Tracking_Trajectory, u_lim_method);
        end

        u_all(:, s) = unew(:, 1);
        if reset_lambda; lambda = lambda_i; dlambda = dlambda_i; end

        % Simulate true system
        [y_obs, xa_sim] = simulate_system(1, xa_true_traj(:,s), u_all(:,s), ...
                                           c_vec, sqrtR, sqrtQ, ...
                                           augment_states_in_filter, 0, ...
                                           build_u_lims(nu), u_lim_method);
        xa_true_traj(:, s+1) = xa_sim(:, end);

        % SPKF update
        xa_hat_filt = [x_hat_traj(:,s); p_hat_traj(:,s)];

        % FIX 4: Enforce Q floors to prevent parameter covariance explosion
        Pw_f = Pw(:,:,1);
        Pw_f(1:nx,    1:nx)    = max(Pw_f(1:nx,1:nx),    1e-4*eye(nx));
        Pw_f(nx+1:end,nx+1:end)= max(Pw_f(nx+1:end,nx+1:end), 1e-5*eye(np_p));

        Dyn_F  = @(dt_f, xk, wk) DiscreteStateDynamics(dt_f, xk, u_all(:,s), ...
                     c_vec, wk, 0, augment_states_in_filter, 1, 0, ...
                     build_u_lims(nu), u_lim_method);
        Meas_F = @(xk, vk) Measurement(1, xk, u_all(:,s), c_vec, vk, 0, ...
                     augment_states_in_filter, 1);

        [xa_filt_all, P_filt] = SPKF_function(1, Dyn_F, Meas_F, y_obs(end,:)', ...
                                            xa_hat_filt, zeros(nx+np_p,1), ...
                                            zeros(ny,1), cov_xa, ...
                                            Pw_f * keep_dyn_noise_filter, ...
                                            Pv(:,:,1) * keep_meas_noise_filter, ...
                                            h_spkf, AdditiveNoise);

        % SPKF_function returns x_hat_all as [iterations x n_states] matrix
        % and P_x as the final [n_states x n_states] covariance.
        % Extract the last (only) row and transpose to get a column vector.
        xa_filt = xa_filt_all(end, :)';   % [nx+np_p x 1]

        % FIX 4: Floor the full updated covariance on both blocks
        P_filt = 0.5*(P_filt + P_filt');
        P_filt(1:nx,    1:nx)    = max(P_filt(1:nx,1:nx),    1e-5*eye(nx));
        P_filt(nx+1:end,nx+1:end)= max(P_filt(nx+1:end,nx+1:end), 1e-6*eye(np_p));

        x_hat_traj(:, s+1) = xa_filt(1:nx);
        p_hat_traj(:, s+1) = max(0.01, xa_filt(nx+1:end));  % keep positive
        cov_xa             = P_filt;

        % Track convergence
        cov_p_s = P_filt(nx+1:end, nx+1:end);
        param_uncertainty(p, s+1) = trace(cov_p_s);
        param_error(p, s+1)       = norm(p_hat_traj(:,s+1) - xi_true_p);

        % Warm start
        u_bar = [unew(:,2:end), unew(:,end)];
    end

    % MFG
    x_final    = xa_true_traj(1:2, cfg_data.n_sessions);
    init_score = (X_eval(1,1,p) + X_eval(2,1,p)) / 2;
    final_score= (x_final(1) + x_final(2)) / 2;
    ceiling_gap= max(swat_p - init_score, 0.01);
    gain_ilqg(p) = (final_score - init_score) / ceiling_gap;

    fprintf('  MFG=%.3f  param_err_final=%.4f\n', gain_ilqg(p), param_error(p,end));
end

%% ── 5. Statistical comparison ────────────────────────────────────────────

mfg_ilqg     = mean(gain_ilqg);
mfg_baseline = mean(gain_baseline);
sd_ilqg      = std(gain_ilqg);
sd_baseline  = std(gain_baseline);

[p_val, ~]   = ranksum(gain_ilqg, gain_baseline, 'tail', 'right');
d_cohen      = (mfg_ilqg - mfg_baseline) / ...
               sqrt((var(gain_ilqg) + var(gain_baseline)) / 2);

fprintf('\n════════════════════════════════════════════════\n');
fprintf('  DUAL iLQG vs CLINICAL BASELINE\n');
fprintf('════════════════════════════════════════════════\n');
fprintf('  Dual iLQG MFG : %.4f ± %.4f\n', mfg_ilqg, sd_ilqg);
fprintf('  Baseline MFG  : %.4f ± %.4f\n', mfg_baseline, sd_baseline);
fprintf('  Improvement   : %.1f%%\n', 100*(mfg_ilqg - mfg_baseline)/abs(mfg_baseline));
fprintf('  Wilcoxon p    : %.4f\n', p_val);
fprintf('  Cohen''s d     : %.2f\n', d_cohen);
fprintf('  Param. error  : %.4f → %.4f (sessions 1 → %d)\n', ...
        mean(param_error(:,1)), mean(param_error(:,end)), cfg_data.n_sessions);
fprintf('════════════════════════════════════════════════\n\n');

%% ── 6. Fig 6A: MFG box plot ──────────────────────────────────────────────

fig6a = figure('Name','Fig 6A - Dual iLQG vs Baseline','Position',[100,100,600,520]);

data_box   = [gain_baseline, gain_ilqg];
labels_box = {'Clinical baseline', 'Dual iLQG'};
colors_box = [0.7 0.7 0.7; 0.2 0.5 0.8];

hold on;
for k = 1:2
    bp = boxplot(data_box(:,k), 'Positions', k, 'Widths', 0.55, 'Symbol', 'o', ...
                 'Colors', colors_box(k,:), 'MedianStyle', 'line');
    set(bp, 'LineWidth', 1.5);
    patch([k-0.275, k+0.275, k+0.275, k-0.275], ...
          [prctile(data_box(:,k),25)*[1 1], prctile(data_box(:,k),75)*[1 1]], ...
          colors_box(k,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end

set(gca, 'XTick', 1:2, 'XTickLabel', labels_box, 'FontSize', 12, 'YGrid', 'on');
ylabel('Normalised Functional Gain (MFG)', 'FontSize', 13);
title({'Dual iLQG vs Clinical Best-Practice Baseline'; ...
       sprintf('N = %d evaluation patients', n_eval)}, ...
      'FontSize', 12, 'FontWeight', 'bold');

annotation('textbox', [0.55, 0.72, 0.38, 0.20], 'String', ...
    {sprintf('Wilcoxon rank-sum (one-sided):'), ...
     sprintf('p = %.4f', p_val), ...
     sprintf('Cohen''s d = %.2f', d_cohen), ...
     sprintf('Improvement: %.1f%%', 100*(mfg_ilqg-mfg_baseline)/abs(mfg_baseline))}, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w', ...
    'EdgeColor', [0.5,0.5,0.5], 'FontSize', 10);

saveas(fig6a, fullfile(out_dir, 'Fig6A_ilqg_vs_baseline.png'));
saveas(fig6a, fullfile(out_dir, 'Fig6A_ilqg_vs_baseline.svg'));

%% ── 7. Fig 6B: Parameter uncertainty convergence ─────────────────────────

fig6b = figure('Name','Fig 6B - Parameter Convergence','Position',[150,150,700,420]);

sessions = 1:cfg_data.n_sessions;
mean_unc = mean(param_uncertainty, 1);
std_unc  = std(param_uncertainty,  0, 1);
mean_err = mean(param_error, 1);
std_err  = std(param_error,  0, 1);

yyaxis left;
errorbar(sessions, mean_unc, std_unc, 'b-o', 'LineWidth', 2, ...
         'MarkerSize', 6, 'MarkerFaceColor', 'b', 'CapSize', 4);
ylabel('Parameter uncertainty: trace(\Sigma_p)', 'FontSize', 12, 'Color', 'b');
set(gca, 'YColor', 'b');

yyaxis right;
errorbar(sessions, mean_err, std_err, 'r--s', 'LineWidth', 2, ...
         'MarkerSize', 6, 'MarkerFaceColor', 'r', 'CapSize', 4);
ylabel('Parameter error: ||\xi_hat - \xi_{true}||', 'FontSize', 12, 'Color', 'r');
set(gca, 'YColor', 'r');

xlabel('Session (weeks)', 'FontSize', 13);
title({'Dual iLQG: Implicit Exploration-Exploitation Transition'; ...
       'Parameter uncertainty converges as model is identified'}, ...
      'FontSize', 12, 'FontWeight', 'bold');
legend({'Uncertainty (trace \Sigma_p)', 'Estimation error ||\xi||'}, ...
       'Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 12, 'XGrid', 'on', 'YGrid', 'on');
xticks(1:cfg_data.n_sessions);

[~, trans_idx] = min(abs(mean_unc - mean_unc(1)/2));
xline(trans_idx, 'k--', sprintf('50%% uncertainty\nreduction: session %d', trans_idx), ...
      'LineWidth', 1.2, 'FontSize', 9, 'LabelHorizontalAlignment', 'right');

saveas(fig6b, fullfile(out_dir, 'Fig6B_param_convergence.png'));
saveas(fig6b, fullfile(out_dir, 'Fig6B_param_convergence.svg'));

%% ── 8. Save results ──────────────────────────────────────────────────────

save(fullfile(out_dir, 'ilqg_cohort_results.mat'), ...
     'gain_ilqg', 'gain_baseline', ...
     'mfg_ilqg', 'mfg_baseline', 'sd_ilqg', 'sd_baseline', ...
     'p_val', 'd_cohen', ...
     'param_uncertainty', 'param_error', 'n_eval');

fprintf('Results saved to %s/\n', out_dir);

%% =========================================================================
%% Local helpers
%% =========================================================================

function u_lims = build_u_lims(nu)
%BUILD_U_LIMS  Build [nu x 2] control limits matrix.
u_lims = [zeros(nu,1), ones(nu,1)];
end
