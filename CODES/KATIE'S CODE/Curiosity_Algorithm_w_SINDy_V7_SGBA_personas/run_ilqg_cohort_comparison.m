%% =========================================================================
%  run_ilqg_cohort_comparison.m
%
%  Demonstrates the value of dual iLQG over adaptive iLQG and clinical
%  baseline on the 100-patient evaluation cohort.
%
% =========================================================================

clear; clc; close all;
clear global SINDY_MODEL;
rng(42);

%% ── 0. Paths ─────────────────────────────────────────────────────────────

addpath('../sindy_paper');   % SINDy pipeline functions
out_dir = 'results/cohort_comparison';
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% ── 1. Load or generate the 500-patient dataset ──────────────────────────

dataset_path = '../sindy_paper/results/paper/paper_results.mat';

fprintf('=== Loading evaluation cohort ===\n');

% Regenerate the same dataset used in run_paper_results.m (rng seed 42)
cfg = struct();
cfg.n_patients      = 500;
cfg.n_sessions      = 12;
cfg.noise_std       = 0.05;
cfg.poly_order      = 2;
cfg.include_trig    = false;
cfg.include_cross   = true;
cfg.lambda          = 0.05;
cfg.max_iter        = 100;
cfg.tol             = 1e-6;
cfg.nmae_threshold  = 1.0;
cfg.max_terms       = 15;
cfg.min_terms       = 3;
cfg.protected_terms = {'a1', 'x6*a1', 'x3*a1'};
cfg.val_fraction    = 0.2;
cfg.n_lambda        = 20;
cfg.verbose         = false;

full_dataset = generate_synthetic_dataset(cfg);

n_eval   = round(cfg.val_fraction * cfg.n_patients);   % 100
eval_idx = (cfg.n_patients - n_eval + 1) : cfg.n_patients;
train_pool = setdiff(1:cfg.n_patients, eval_idx);

X_eval    = full_dataset.X(:, :, eval_idx);
swat_eval = full_dataset.swat_ceiling(eval_idx);

fprintf('  Evaluation cohort: %d patients\n', n_eval);

%% ── 2. Build oracle SINDy model (N = 100 training patients) ──────────────

fprintf('\n=== Building SINDy model (N=100) ===\n');
rng(42);  % reproducible subsample
N_train = 100;
tr_idx  = train_pool(randperm(length(train_pool), N_train));
X_tr    = full_dataset.X(:, :, tr_idx);
A_tr    = full_dataset.A(:, :, tr_idx);

[Theta, dX, lib_labels, protected_idx] = build_sindy_library(X_tr, A_tr, cfg);
lambda_opt = cross_validate_lambda(Theta, dX, cfg, protected_idx);
cfg.lambda  = lambda_opt;

Xi = stlsq(Theta, dX, lambda_opt, cfg.max_iter, cfg.tol, false, ...
            cfg.max_terms, protected_idx, cfg.min_terms);
[Xi, ~] = prune_sindy_bic(Xi, Theta, dX, lib_labels, ...
                           full_dataset.state_names, false, protected_idx);
Xi(:, 3:6) = 0;  % zero static state columns

fprintf('  SINDy model: %d non-zero terms\n', nnz(Xi));

%% ── 3. iLQG shared settings ──────────────────────────────────────────────

% Dimensions
nx = 6;   nu = 3;   ny = 2;   nv = ny;
np_max  = 2;   % keep small for tractability across 100 patients

% iLQG options (quiet, fast)
lambda_i   = 1;   dlambda_i  = 1;
regType    = 3;
max_iter_ilqg = 80;    % reduced for cohort run
first_iter    = 120;
horizon_length = 6;
u_lims     = repmat([0, 1], nu, 1);
u_lim_method = 2;
h_spkf     = sqrt(3);
AdditiveNoise = 0;
keep_dyn_noise_filter  = 1;
keep_meas_noise_filter = 1;
R_cov = 1e-3 * eye(ny);
sqrtR = chol(R_cov, 'lower');

%% ── 4. Clinical baseline gains ───────────────────────────────────────────

fprintf('\n=== Computing clinical baseline gains ===\n');
ais_protocol = [0.25, 0.40, 0.50, 0.60;
                0.50, 0.55, 0.50, 0.70;
                0.75, 0.70, 0.50, 0.80;
                1.00, 0.80, 0.50, 0.85];

gain_baseline = zeros(n_eval, 1);
for p = 1:n_eval
    x       = X_eval(:, 1, p);
    ceil_p  = swat_eval(p);
    ais_val = x(3);
    [~, ar] = min(abs(ais_protocol(:,1) - ais_val));
    a1      = ais_protocol(ar, 2);

    for s = 1:cfg.n_sessions - 1
        sat1 = max(0, ceil_p - x(1));
        dx1  = (-0.05*x(1) + 0.30*x(1)*x(3) + 0.25*a1 + 0.20*a1*x(3) + 0.10*x(6)*a1)*sat1;
        sat2 = max(0, ceil_p - x(2));
        dx2  = (-0.04*x(2) + 0.22*x(2)*x(3) + 0.18*a1 + 0.15*x(6)*a1)*sat2;
        x(1) = max(0, min(1, x(1) + dx1 + cfg.noise_std*randn()));
        x(2) = max(0, min(1, x(2) + dx2 + cfg.noise_std*randn()));
    end
    init  = (X_eval(1,1,p) + X_eval(2,1,p)) / 2;
    final = (x(1) + x(2)) / 2;
    gain_baseline(p) = (final - init) / max(ceil_p - init, 0.01);
end
fprintf('  Baseline MFG: %.4f ± %.4f\n', mean(gain_baseline), std(gain_baseline));

%% ── 5. Run dual iLQG and adaptive iLQG on all eval patients ──────────────

gain_dual     = zeros(n_eval, 1);
gain_adaptive = zeros(n_eval, 1);
p_err_dual    = zeros(n_eval, cfg.n_sessions);  % parameter RMSE per session
p_err_adaptive= zeros(n_eval, cfg.n_sessions);

fprintf('\n=== Running dual iLQG and adaptive iLQG on %d patients ===\n', n_eval);
fprintf('    (this may take several minutes)\n\n');

for p = 1:n_eval

    if mod(p, 10) == 0
        fprintf('  Patient %d / %d  (dual MFG so far: %.3f)\n', p, n_eval, ...
                mean(gain_dual(1:p-1), 'omitnan'));
    end

    x0      = X_eval(:, 1, p);
    ceil_p  = swat_eval(p);

    % Identify the np uncertain parameters for this patient
    [t_idx, s_idx] = find(Xi(:, 1:2) ~= 0);
    n_unc   = min(np_max, length(t_idx));
    xi_idx  = [t_idx(1:n_unc), s_idx(1:n_unc)];
    np      = n_unc;
    nw      = nx + np;

    % True parameter values
    xi_true = zeros(np, 1);
    for k = 1:np
        xi_true(k) = Xi(xi_idx(k,1), xi_idx(k,2));
    end

    %% Run both modes in a loop
    for mode = 1:2
        % mode 1: dual iLQG (augment_states_in_iLQG = 1)
        % mode 2: adaptive iLQG (augment_states_in_iLQG = 0)

        augment_ilqg   = (mode == 1);
        augment_filter = 1;  % always estimate parameters in filter

        % Setup global SINDY_MODEL
        global SINDY_MODEL;
        SINDY_MODEL.Xi           = Xi;
        SINDY_MODEL.lib_cfg      = cfg;
        SINDY_MODEL.swat_ceiling = ceil_p;
        SINDY_MODEL.xi_idx       = xi_idx;
        SINDY_MODEL.F_dyn        = 1e-3;
        SINDY_MODEL.F_param      = 1e-4;
        c_vec = [ceil_p; 1e-3; 1e-4];

        % Initial estimates
        cov_X = 0.01;
        cov_P = diag((0.5 * abs(xi_true) + 0.01).^2);
        x_hat = max(0, min(1, x0 + sqrt(cov_X)*randn(nx,1)));
        p_hat = xi_true + chol(cov_P,'lower')*randn(np,1);
        cov_xa = blkdiag(cov_X*eye(nx), cov_P);

        % Augmented true state
        xa_true_k = [x0; xi_true];

        ix = 1:nx;  ip = (1:np)+nx;

        % Control initialisation
        l_gain = 0.5 * ones(nu, horizon_length);
        L_gain = zeros(nu, nx + np*augment_ilqg, horizon_length);
        u_bar  = 0.5 * ones(nu, horizon_length);
        T_rem  = horizon_length * ones(1, cfg.n_sessions + 1);

        lambda  = lambda_i;
        dlambda = dlambda_i;

        x_traj = zeros(nx, cfg.n_sessions);
        p_traj = zeros(np, cfg.n_sessions);
        x_traj(:, 1) = x_hat;
        p_traj(:, 1) = p_hat;

        Q_cov_aug = 1e-3 * eye(nx + np);
        sqrtQ_aug = chol(Q_cov_aug, 'lower');

        for iter = 1:cfg.n_sessions - 1

            % iLQG inner loop
            try
                [~, unew, l_gain, L_gain, ~, ~, ~, Pw, Pv, ~] = ...
                    iLQG_function(T_rem(iter), 1, x_hat, l_gain, L_gain, u_bar, ...
                                  lambda, dlambda, c_vec, p_hat, cov_xa, ...
                                  augment_ilqg, regType, u_lims, ny, nv, nw, ...
                                  max_iter_ilqg, 0, 0, 0, [], u_lim_method);
            catch ME
                % If iLQG fails entirely, fall back to fixed intensity
                warning('Patient %d session %d iLQG error: %s', p, iter, ME.message);
                unew = 0.5 * ones(nu, horizon_length);
                Pw   = 1e-3 * eye(nx+np);  Pw = repmat(Pw, 1, 1, horizon_length);
                Pv   = 1e-3 * eye(ny);     Pv = repmat(Pv, 1, 1, horizon_length);
            end

            u_applied = unew(:, 1);

            % Simulate ground-truth ODE (one session)
            x_curr = xa_true_k(1:nx);
            a1 = (u_lims(1,2)-u_lims(1,1))/2*tanh(u_applied(1)) + (u_lims(1,2)+u_lims(1,1))/2;

            sat1 = max(0, ceil_p - x_curr(1));
            dx1  = (-0.05*x_curr(1) + 0.30*x_curr(1)*x_curr(3) + 0.25*a1 + ...
                     0.20*a1*x_curr(3) + 0.10*x_curr(6)*a1)*sat1;
            sat2 = max(0, ceil_p - x_curr(2));
            dx2  = (-0.04*x_curr(2) + 0.22*x_curr(2)*x_curr(3) + ...
                     0.18*a1 + 0.15*x_curr(6)*a1)*sat2;

            x_next    = x_curr;
            x_next(1) = max(0, min(1, x_curr(1) + dx1 + cfg.noise_std*randn()));
            x_next(2) = max(0, min(1, x_curr(2) + dx2 + cfg.noise_std*randn()));
            xa_true_k = [x_next; xi_true];

            % Observation
            y_obs = x_next(1:2) + 0.03*randn(2,1);

            % SPKF update
            xa_hat_filt = [x_hat; p_hat];
            Pw_f = Pw(:,:,1);
            Pv_f = Pv(:,:,1);
            Pw_f(nx+1:end, nx+1:end) = max(Pw_f(nx+1:end,nx+1:end), ...
                                            1e-4*eye(np));

            Dyn_F  = @(dt_f, xk, wk) DiscreteStateDynamics(dt_f, xk, u_applied, ...
                         c_vec, wk, 0, augment_filter, 1, 0, u_lims, u_lim_method);
            Meas_F = @(xk, vk) Measurement(1, xk, u_applied, c_vec, vk, 0, ...
                                             augment_filter, 1);

            try
                [xa_filt, P_filt] = SPKF_function(1, Dyn_F, Meas_F, y_obs', ...
                                                    xa_hat_filt, zeros(nx+np,1), ...
                                                    zeros(ny,1), cov_xa, ...
                                                    Pw_f*keep_dyn_noise_filter, ...
                                                    Pv_f*keep_meas_noise_filter, ...
                                                    h_spkf, AdditiveNoise);
            catch
                xa_filt = xa_hat_filt;
                P_filt  = cov_xa;
            end

            x_hat  = xa_filt(ix);
            p_hat  = xa_filt(ip);
            cov_xa = P_filt;

            x_traj(:, iter+1) = x_hat;
            p_traj(:, iter+1) = p_hat;

            % Warm start
            u_bar = [unew(:, 2:end), unew(:, end)];
            lambda  = lambda_i;
            dlambda = dlambda_i;

            % Parameter error (RMSE across uncertain params)
            if mode == 1
                p_err_dual(p, iter)     = sqrt(mean((p_hat - xi_true).^2));
            else
                p_err_adaptive(p, iter) = sqrt(mean((p_hat - xi_true).^2));
            end

        end  % sessions

        % Compute MFG from true trajectory
        x_final  = xa_true_k(1:2);
        init_sc  = (X_eval(1,1,p) + X_eval(2,1,p)) / 2;
        final_sc = (x_final(1) + x_final(2)) / 2;
        mfg = (final_sc - init_sc) / max(ceil_p - init_sc, 0.01);

        if mode == 1
            gain_dual(p)     = mfg;
        else
            gain_adaptive(p) = mfg;
        end

    end  % mode loop

end  % patient loop

%% ── 6. Statistics ────────────────────────────────────────────────────────

fprintf('\n════════════════════════════════════════════════════════\n');
fprintf('  COHORT COMPARISON RESULTS\n');
fprintf('════════════════════════════════════════════════════════\n\n');
fprintf('  Policy              MFG mean ± SD\n');
fprintf('  ─────────────────────────────────\n');
fprintf('  Dual iLQG         : %.4f ± %.4f\n', mean(gain_dual),     std(gain_dual));
fprintf('  Adaptive iLQG     : %.4f ± %.4f\n', mean(gain_adaptive), std(gain_adaptive));
fprintf('  Clinical baseline : %.4f ± %.4f\n', mean(gain_baseline), std(gain_baseline));

% Wilcoxon rank-sum tests (one-sided: dual > adaptive, dual > baseline)
[p_dual_vs_adapt, ~]   = ranksum(gain_dual, gain_adaptive,  'tail','right');
[p_dual_vs_base,  ~]   = ranksum(gain_dual, gain_baseline,  'tail','right');
[p_adapt_vs_base, ~]   = ranksum(gain_adaptive, gain_baseline, 'tail','right');

d_dual_vs_adapt   = cohens_d_local(gain_dual, gain_adaptive);
d_dual_vs_base    = cohens_d_local(gain_dual, gain_baseline);
d_adapt_vs_base   = cohens_d_local(gain_adaptive, gain_baseline);

fprintf('\n  Wilcoxon rank-sum (one-sided, proposed > comparator):\n');
fprintf('  Dual vs Adaptive  : p=%.4f, d=%.2f\n', p_dual_vs_adapt, d_dual_vs_adapt);
fprintf('  Dual vs Baseline  : p=%.4f, d=%.2f\n', p_dual_vs_base,  d_dual_vs_base);
fprintf('  Adaptive vs Baseline: p=%.4f, d=%.2f\n', p_adapt_vs_base, d_adapt_vs_base);

% Parameter convergence
mean_p_err_dual     = mean(p_err_dual,     1);
mean_p_err_adaptive = mean(p_err_adaptive, 1);
sessions_vec = 1:cfg.n_sessions;

% Session at which parameter RMSE first drops below 20% of initial
thresh = 0.2 * mean_p_err_dual(1);
conv_dual_sess     = find(mean_p_err_dual     < thresh, 1, 'first');
conv_adaptive_sess = find(mean_p_err_adaptive < thresh, 1, 'first');
fprintf('\n  Parameter convergence (RMSE < 20%% of initial):\n');
if ~isempty(conv_dual_sess)
    fprintf('    Dual iLQG    : session %d\n', conv_dual_sess);
else
    fprintf('    Dual iLQG    : did not converge within horizon\n');
end
if ~isempty(conv_adaptive_sess)
    fprintf('    Adaptive iLQG: session %d\n', conv_adaptive_sess);
else
    fprintf('    Adaptive iLQG: did not converge within horizon\n');
end

%% ── 7. Figure 6: MFG box plots ───────────────────────────────────────────

fig6 = figure('Name','Fig 6 - iLQG Cohort Comparison','Position',[100,100,720,520]);
data_box = [gain_baseline, gain_adaptive, gain_dual];
labels_6 = {'Clinical baseline','Adaptive iLQG','Dual iLQG'};
colors_6  = [0.7 0.7 0.7; 0.3 0.6 0.9; 0.2 0.7 0.4];

hold on;
for k = 1:3
    bp = boxplot(data_box(:,k), 'Positions', k, 'Widths', 0.55, 'Symbol','o', ...
                 'Colors', colors_6(k,:), 'MedianStyle','line');
    set(bp,'LineWidth',1.5);
    patch([k-0.275 k+0.275 k+0.275 k-0.275], ...
          [prctile(data_box(:,k),25)*[1 1], prctile(data_box(:,k),75)*[1 1]], ...
          colors_6(k,:), 'FaceAlpha', 0.25, 'EdgeColor','none');
end
set(gca,'XTick',1:3,'XTickLabel',labels_6,'FontSize',11,'YGrid','on');
ylabel('Normalised Functional Gain (MFG)','FontSize',12);
title({'Dual iLQG vs Adaptive iLQG vs Clinical Baseline'; ...
       '100-patient evaluation cohort'},'FontSize',12,'FontWeight','bold');

annotation('textbox',[0.57,0.68,0.40,0.22],'String', ...
    {sprintf('Dual vs Adaptive: p=%.4f, d=%.2f', p_dual_vs_adapt, d_dual_vs_adapt), ...
     sprintf('Dual vs Baseline: p=%.4f, d=%.2f', p_dual_vs_base,  d_dual_vs_base), ...
     sprintf('Adaptive vs Baseline: p=%.4f, d=%.2f', p_adapt_vs_base, d_adapt_vs_base)}, ...
    'FitBoxToText','on','BackgroundColor','w','EdgeColor',[0.5 0.5 0.5],'FontSize',9);

saveas(fig6, fullfile(out_dir,'Fig6_ilqg_cohort_comparison.png'));
saveas(fig6, fullfile(out_dir,'Fig6_ilqg_cohort_comparison.svg'));

%% ── 8. Figure 7: Parameter convergence ───────────────────────────────────

fig7 = figure('Name','Fig 7 - Parameter Convergence','Position',[150,150,760,440]);
sd_dual     = std(p_err_dual,     0, 1);
sd_adaptive = std(p_err_adaptive, 0, 1);

errorbar(sessions_vec, mean_p_err_dual,     sd_dual,     'g-o','LineWidth',2, ...
         'MarkerSize',6,'MarkerFaceColor','g','CapSize',4, ...
         'DisplayName','Dual iLQG');
hold on;
errorbar(sessions_vec, mean_p_err_adaptive, sd_adaptive, 'b-s','LineWidth',2, ...
         'MarkerSize',6,'MarkerFaceColor','b','CapSize',4, ...
         'DisplayName','Adaptive iLQG');

yline(thresh,'k--',sprintf('20%% threshold (%.4f)', thresh), ...
      'LineWidth',1.2,'FontSize',10);

if ~isempty(conv_dual_sess)
    xline(conv_dual_sess,'g--', sprintf('Dual converges: session %d', conv_dual_sess), ...
          'LineWidth',1.2,'FontSize',9,'Color',[0.2 0.7 0.4]);
end
if ~isempty(conv_adaptive_sess)
    xline(conv_adaptive_sess,'b--', sprintf('Adaptive converges: session %d', conv_adaptive_sess), ...
          'LineWidth',1.2,'FontSize',9,'Color',[0.3 0.6 0.9]);
end

set(gca,'FontSize',12,'YGrid','on','XGrid','on');
xlabel('Session','FontSize',13);
ylabel('Mean parameter RMSE','FontSize',13);
title({'SINDy Parameter Convergence: Dual iLQG vs Adaptive iLQG'; ...
       'Mean ± SD across 100-patient evaluation cohort'},'FontSize',12,'FontWeight','bold');
legend('Location','northeast','FontSize',11);
xlim([1 cfg.n_sessions]);  box on;

saveas(fig7, fullfile(out_dir,'Fig7_parameter_convergence.png'));
saveas(fig7, fullfile(out_dir,'Fig7_parameter_convergence.svg'));

%% ── 9. Save ──────────────────────────────────────────────────────────────

save(fullfile(out_dir,'cohort_comparison_results.mat'), ...
    'gain_dual','gain_adaptive','gain_baseline', ...
    'p_err_dual','p_err_adaptive', ...
    'p_dual_vs_adapt','p_dual_vs_base','p_adapt_vs_base', ...
    'd_dual_vs_adapt','d_dual_vs_base','d_adapt_vs_base', ...
    'mean_p_err_dual','mean_p_err_adaptive', ...
    'conv_dual_sess','conv_adaptive_sess');

fprintf('\nAll results saved to %s/\n', out_dir);
fprintf('════════════════════════════════════════════════════════\n\n');

%% ── Local helpers ────────────────────────────────────────────────────────

function d = cohens_d_local(x, y)
d = (mean(x) - mean(y)) / sqrt((var(x) + var(y)) / 2);
end
