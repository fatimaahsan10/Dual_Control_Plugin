%% =========================================================================
%  main_sindy_pipeline.m
%  SINDy-based dynamic model generation for SCI rehabilitation planning
%
%  Pipeline steps:
%    1. Generate (or load) a synthetic SCI rehabilitation dataset
%    2. Build the SINDy candidate library Psi(x)
%    3. Compute state derivatives (via finite differences or smoothing)
%    4. Run STLSQ to identify sparse coefficients Xi
%    5. Validate model via forward simulation (NMAE threshold < 1.0)
%    6. Report identified equations and Fisher information matrix
%
%  Author : Katie Campbell, UNB ECE
%  Date   : May 2026
% =========================================================================

clear; clc; close all;
rng(42);   % reproducibility

%% ── 0. Configuration ─────────────────────────────────────────────────────

cfg = struct();

% Dataset
cfg.n_patients      = 100;    % number of simulated patients for training
cfg.n_sessions      = 12;     % weekly sessions (12-week inpatient horizon)
cfg.noise_std       = 0.05;   % Gaussian observation noise std (normalised)

% SINDy library
cfg.poly_order      = 2;      % max polynomial degree in Psi(x)
cfg.include_trig    = false;  % include sin/cos terms in library
cfg.include_cross   = true;   % include pairwise cross-products

% STLSQ
cfg.lambda          = 0.05;   % sparsity threshold (tuned via cross-validation)
cfg.max_iter        = 100;    % max STLSQ iterations
cfg.tol             = 1e-6;   % convergence tolerance
cfg.max_terms       = 22;     % hard cap on active terms per state equation
                              % (raised from 15 to fit the 13 therapy terms +
                              %  key state/interaction terms)
                              % (applied after STLSQ; BIC pruning refines further)
cfg.min_terms       = 3;      % minimum terms to retain per dynamic state equation
                              % (prevents over-pruning; must include at least one
                              %  action term for the controller to optimise)

% Terms that must survive thresholding regardless of coefficient magnitude.
% Format: cell array of library label substrings to protect.
% The 13 linear therapy terms a1..a13 are auto-protected inside
% build_sindy_library. Here we additionally protect two clinically mandated
% action-state interactions: x3*a3 (AIS x strengthening) and x6*a8
% (caregiver x gait training).
cfg.protected_terms = {'x3*a3', 'x6*a8'};

% Validation
cfg.nmae_threshold  = 1.0;    % accept model if NMAE < this value
cfg.val_fraction    = 0.2;    % fraction of patients held out for validation
cfg.n_lambda        = 20;     % number of lambda values to sweep in CV

% Output
cfg.verbose         = true;
cfg.save_results    = true;
cfg.output_dir      = 'results';

if cfg.save_results && ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

%% ── 1. Generate synthetic dataset ────────────────────────────────────────

if cfg.verbose
    fprintf('\n=== Step 1: Generating synthetic SCI dataset ===\n');
end

dataset = generate_synthetic_dataset(cfg);

fprintf('  Patients : %d\n', cfg.n_patients);
fprintf('  Sessions : %d\n', cfg.n_sessions);
fprintf('  State dim: %d (%s)\n', dataset.n_states, ...
        strjoin(dataset.state_names, ', '));

%% ── 2. Split train / validation ──────────────────────────────────────────

n_val   = round(cfg.val_fraction * cfg.n_patients);
n_train = cfg.n_patients - n_val;
idx     = randperm(cfg.n_patients);

train_idx = idx(1:n_train);
val_idx   = idx(n_train+1:end);

X_train  = dataset.X(:, :, train_idx);   % [n_states x n_sessions x n_train]
A_train  = dataset.A(:, :, train_idx);   % [n_actions x n_sessions x n_train]
Y_train  = dataset.Y(:, :, train_idx);   % [n_outcomes x n_sessions x n_train]

X_val    = dataset.X(:, :, val_idx);
A_val    = dataset.A(:, :, val_idx);
Y_val    = dataset.Y(:, :, val_idx);

%% ── 3. Build SINDy library ───────────────────────────────────────────────

if cfg.verbose
    fprintf('\n=== Step 2: Building SINDy candidate library ===\n');
end

% Stack all training trajectories into one large data matrix
% Each row = one observation [state; action] at one time step
[Theta, dX, lib_labels, protected_idx] = build_sindy_library(X_train, A_train, cfg);

fprintf('  Library size: %d candidate terms\n', size(Theta, 2));
fprintf('  Observations: %d (across %d patients x %d sessions)\n', ...
        size(Theta,1), n_train, cfg.n_sessions - 1);

%% ── 4. Cross-validate lambda and run STLSQ ───────────────────────────────

if cfg.verbose
    fprintf('\n=== Step 3: Cross-validating sparsity threshold lambda ===\n');
end

lambda_opt = cross_validate_lambda(Theta, dX, cfg, protected_idx);
fprintf('  Optimal lambda: %.4f\n', lambda_opt);

cfg.lambda = lambda_opt;

if cfg.verbose
    fprintf('\n=== Step 4: Running STLSQ sparse regression ===\n');
end

Xi = stlsq(Theta, dX, cfg.lambda, cfg.max_iter, cfg.tol, cfg.verbose, ...
           cfg.max_terms, protected_idx, cfg.min_terms);

fprintf('  Non-zero coefficients: %d / %d\n', nnz(Xi), numel(Xi));

%% ── 4b. BIC pruning ──────────────────────────────────────────────────────

if cfg.verbose
    fprintf('\n=== Step 4b: BIC post-selection pruning ===\n');
end

[Xi, bic_history] = prune_sindy_bic(Xi, Theta, dX, lib_labels, ...
                                     dataset.state_names, cfg.verbose, protected_idx);

fprintf('  Non-zero after BIC pruning: %d / %d  (sparsity: %.1f%%)\n', ...
        nnz(Xi), numel(Xi), 100*(1 - nnz(Xi)/numel(Xi)));

%% ── 4c. Zero static state equations ─────────────────────────────────────
% State variables x3–x6 (AIS, Age, DPI, Caregiver) are fixed patient
% attributes — they do not evolve during rehabilitation. Any non-zero
% coefficients in their equations are spurious fits to measurement noise.
% Zeroing them ensures the iLQG controller only optimises true dynamics.

static_states = 3 : dataset.n_states;   % indices of non-dynamic states
Xi(:, static_states) = 0;

if cfg.verbose
    fprintf('\n  Zeroed static state equations (x3–x6): %d / %d coefficients remain\n', ...
            nnz(Xi), numel(Xi));
end

%% ── 5. Validate model ────────────────────────────────────────────────────

if cfg.verbose
    fprintf('\n=== Step 5: Validating model via forward simulation ===\n');
end

[nmae, rmse, sim_trajectories] = validate_sindy_model( ...
    Xi, X_val, A_val, Y_val, lib_labels, cfg);

fprintf('  Validation NMAE : %.4f  (threshold: %.1f)\n', nmae, cfg.nmae_threshold);
fprintf('  Validation RMSE : %.4f\n', rmse);

if nmae < cfg.nmae_threshold
    fprintf('  [PASS] Model accepted.\n');
else
    fprintf('  [FAIL] Model rejected. Consider adjusting lambda or library.\n');
end

%% ── 6. Compute Fisher information matrix ─────────────────────────────────

if cfg.verbose
    fprintf('\n=== Step 6: Computing Fisher information matrix ===\n');
end

F = compute_fisher_information(Theta, cfg.noise_std);
Sigma = inv(F + 1e-8 * eye(size(F)));   % parameter covariance (regularised)

fprintf('  Condition number of F: %.2e\n', cond(F));

%% ── 7. Report identified equations ───────────────────────────────────────

print_identified_equations(Xi, lib_labels, dataset.state_names, cfg);

%% ── 8. Visualise results ─────────────────────────────────────────────────

plot_results(sim_trajectories, Y_val, dataset, cfg);

%% ── 9. Save outputs ──────────────────────────────────────────────────────

if cfg.save_results
    save(fullfile(cfg.output_dir, 'sindy_model.mat'), ...
         'Xi', 'lib_labels', 'F', 'Sigma', 'nmae', 'rmse', 'cfg', 'dataset');
    fprintf('\nModel saved to %s/sindy_model.mat\n', cfg.output_dir);
end

fprintf('\n=== Pipeline complete ===\n\n');
