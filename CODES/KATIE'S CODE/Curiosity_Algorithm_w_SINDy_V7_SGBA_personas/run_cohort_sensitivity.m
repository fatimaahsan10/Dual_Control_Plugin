%% =========================================================================
%  run_cohort_sensitivity.m
%  Analyse SINDy model accuracy as a function of training cohort size.
%
%  Reproduces the mean prediction error vs. n_patients curve reported in
%  the Current Progress section (Specific Aim 2) of the thesis proposal,
%  and extends it for the manuscript's Data Efficiency metric (Section V-C).
%
%  This script runs the full SINDy pipeline at multiple training cohort
%  sizes and plots NMAE vs. n_patients on a log-scale x-axis.
%
%  Author : Katie Campbell, UNB ECE
% =========================================================================

clear; clc; close all;
rng(42);

%% ── Configuration ────────────────────────────────────────────────────────

cfg = struct();
cfg.n_patients      = 500;    % total pool size
cfg.n_sessions      = 12;
cfg.noise_std       = 0.05;
cfg.poly_order      = 2;
cfg.include_trig    = false;
cfg.include_cross   = true;
cfg.lambda          = 0.05;   % fixed lambda for sensitivity sweep
cfg.max_iter        = 100;
cfg.tol             = 1e-6;
cfg.nmae_threshold  = 1.0;
cfg.max_terms       = 15;     % hard cap per state equation
cfg.min_terms       = 3;      % minimum terms retained per dynamic state
cfg.val_fraction    = 0.2;
cfg.n_lambda        = 20;
cfg.verbose         = false;
cfg.save_results    = true;
cfg.output_dir      = 'results';

if ~exist(cfg.output_dir, 'dir'); mkdir(cfg.output_dir); end

%% ── Generate full dataset once ───────────────────────────────────────────

fprintf('Generating full dataset (%d patients)...\n', cfg.n_patients);
full_dataset = generate_synthetic_dataset(cfg);

% Fixed validation set (last 20% of patients)
n_val_fixed = round(cfg.val_fraction * cfg.n_patients);
val_idx     = (cfg.n_patients - n_val_fixed + 1) : cfg.n_patients;

X_val = full_dataset.X(:, :, val_idx);
A_val = full_dataset.A(:, :, val_idx);
Y_val = full_dataset.Y(:, :, val_idx);

%% ── Cohort size sweep ────────────────────────────────────────────────────

cohort_sizes = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400];
n_repeats    = 5;   % random subsamples per cohort size

nmae_matrix = zeros(length(cohort_sizes), n_repeats);
nnz_matrix  = zeros(length(cohort_sizes), n_repeats);

fprintf('\nRunning cohort size sensitivity sweep...\n');
fprintf('%-12s %-12s %-12s %-12s\n', 'n_train', 'NMAE mean', 'NMAE std', 'nnz mean');
fprintf('%s\n', repmat('-', 1, 50));

available_train = setdiff(1:cfg.n_patients, val_idx);

for ci = 1:length(cohort_sizes)
    n_train = cohort_sizes(ci);

    if n_train > length(available_train)
        nmae_matrix(ci, :) = NaN;
        continue;
    end

    for r = 1:n_repeats
        % Random subsample of training patients
        train_idx = available_train(randperm(length(available_train), n_train));

        X_tr = full_dataset.X(:, :, train_idx);
        A_tr = full_dataset.A(:, :, train_idx);

        % Build library
        [Theta, dX, lib_labels, protected_idx] = build_sindy_library(X_tr, A_tr, cfg);

        % STLSQ (use fixed lambda for fair comparison across cohort sizes)
        Xi = stlsq(Theta, dX, cfg.lambda, cfg.max_iter, cfg.tol, false, ...
                   cfg.max_terms, protected_idx, cfg.min_terms);

        % Validate
        [nmae, ~, ~] = validate_sindy_model(Xi, X_val, A_val, Y_val, ...
                                             lib_labels, cfg);

        nmae_matrix(ci, r) = nmae;
        nnz_matrix(ci, r)  = nnz(Xi);
    end

    fprintf('%-12d %-12.4f %-12.4f %-12.1f\n', ...
            n_train, ...
            mean(nmae_matrix(ci, :), 'omitnan'), ...
            std(nmae_matrix(ci, :), 'omitnan'), ...
            mean(nnz_matrix(ci, :), 'omitnan'));
end

%% ── Plot NMAE vs. cohort size ────────────────────────────────────────────

nmae_mean = mean(nmae_matrix, 2, 'omitnan');
nmae_std  = std(nmae_matrix,  0, 2, 'omitnan');
valid     = ~isnan(nmae_mean);

fig = figure('Name', 'Cohort Size Sensitivity', ...
             'NumberTitle', 'off', 'Position', [100, 100, 750, 500]);

semilogy(cohort_sizes(valid), nmae_mean(valid), 'b-o', ...
         'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'b');
hold on;
% Shaded error band
fill([cohort_sizes(valid), fliplr(cohort_sizes(valid))], ...
     [nmae_mean(valid) + nmae_std(valid); ...
      flipud(nmae_mean(valid) - nmae_std(valid))]', ...
     [0.2, 0.5, 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% NMAE acceptance threshold
yline(cfg.nmae_threshold, 'r--', ...
      sprintf('NMAE threshold = %.1f', cfg.nmae_threshold), ...
      'LineWidth', 1.8, 'FontSize', 10, ...
      'LabelHorizontalAlignment', 'right');

% Find minimum cohort size that crosses threshold
cross_idx = find(nmae_mean(valid) < cfg.nmae_threshold, 1, 'first');
if ~isempty(cross_idx)
    valid_sizes = cohort_sizes(valid);
    xline(valid_sizes(cross_idx), 'g--', ...
          sprintf('DE: N=%d', valid_sizes(cross_idx)), ...
          'LineWidth', 1.5, 'FontSize', 10);
end

xlabel('Number of Training Patients (log scale)', 'FontSize', 12);
ylabel('Normalised MAE (NMAE)', 'FontSize', 12);
title({'SINDy Model Accuracy vs. Training Cohort Size'; ...
       'Mean ± SD across 5 random subsamples'}, ...
      'FontSize', 13, 'FontWeight', 'bold');
legend({'NMAE (mean)', 'NMAE ± SD', 'Acceptance threshold', ...
        'Data Efficiency threshold'}, ...
       'Location', 'northeast', 'FontSize', 10);
set(gca, 'FontSize', 11, 'XScale', 'log');
grid on; box on;
xlim([min(cohort_sizes(valid))*0.8, max(cohort_sizes(valid))*1.2]);

if cfg.save_results
    saveas(fig, fullfile(cfg.output_dir, 'fig_cohort_sensitivity.png'));
    save(fullfile(cfg.output_dir, 'cohort_sensitivity.mat'), ...
         'cohort_sizes', 'nmae_matrix', 'nmae_mean', 'nmae_std', 'cfg');
    fprintf('\nResults saved to %s/\n', cfg.output_dir);
end

fprintf('\nData Efficiency (NMAE < %.1f): ', cfg.nmae_threshold);
if ~isempty(cross_idx)
    fprintf('N = %d patients\n', cohort_sizes(valid_sizes == valid_sizes(cross_idx)));
else
    fprintf('threshold not reached in sweep\n');
end
