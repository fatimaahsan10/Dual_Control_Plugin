%% =========================================================================
%  run_persona_comparison.m
%  Compare SINDy-based iLQG therapy recommendations across all personas,
%  using the 13 WHO rehabilitation control signals (who_therapies.m).
%
%  For each persona this script:
%    1. Generates a persona-specific synthetic dataset (13-therapy dynamics)
%    2. Runs the SINDy pipeline to identify the persona's dynamics
%    3. Runs the curiosity-driven (dual iLQG) controller with that persona's
%       cost weights and the uniform resource-preservation penalty
%    4. Records MFG, final SCIM/BBS, and the mean recommended dose of each
%       of the 13 therapies
%    5. Produces a summary table (top therapies per persona) + figures
%
%  Because each persona has a different per-therapy efficacy profile baked
%  into its identified dynamics, the controller recommends a DIFFERENT,
%  persona-appropriate therapy mix - exactly the intended behaviour.
%
%  Author : Katie Campbell, UNB ECE
%  Date   : June 2026
% =========================================================================

clear; clc; close all;
clear global SINDY_MODEL;
rng(42);

TH = who_therapies();          % 13 therapy definitions (single source)
n_u = TH.n;                    % 13

%% -- Persona list ---------------------------------------------------------

persona_names = { ...
    'high_motivator', ...
    'young_athlete', ...
    'default', ...
    'low_support', ...
    'pain_sensitive', ...
    'frail_elder' ...
};

n_personas = length(persona_names);

%% -- Shared base config ---------------------------------------------------

base_cfg = struct();
base_cfg.n_patients      = 100;
base_cfg.n_sessions      = 12;
base_cfg.noise_std       = 0.05;
base_cfg.poly_order      = 2;
base_cfg.include_trig    = false;
base_cfg.include_cross   = true;
base_cfg.lambda          = 0.05;
base_cfg.max_iter        = 100;
base_cfg.tol             = 1e-6;
base_cfg.nmae_threshold  = 1.0;
base_cfg.max_terms       = 22;   % room for 13 action terms + key interactions
base_cfg.min_terms       = 3;
% Linear therapy terms a1..a13 are auto-protected inside build_sindy_library.
% Additionally protect two clinically mandated interactions:
%   x3*a3 (AIS x strengthening), x6*a8 (caregiver x gait training).
base_cfg.protected_terms = {'x3*a3', 'x6*a8'};
base_cfg.val_fraction    = 0.2;
base_cfg.n_lambda        = 20;
base_cfg.verbose         = false;
base_cfg.save_results    = false;

%% -- Fixed patient initial state ------------------------------------------
% Same patient presented to each persona-tuned planner for a fair comparison.

x0_true_base = [0.25; 0.20; 0.75; 0.45; 0.30; 0.70];
%               SCIM   BBS   AIS   Age   DPI   CG

%% -- Results storage ------------------------------------------------------

results = struct();

for pidx = 1:n_personas
    pname   = persona_names{pidx};
    persona = define_persona(pname);

    fprintf('\n========================================\n');
    fprintf('  Persona %d/%d: %s\n', pidx, n_personas, persona.name);
    fprintf('========================================\n');

    %% Step 1: Generate persona-specific training dataset
    cfg         = base_cfg;
    cfg.persona = persona;

    dataset = generate_synthetic_dataset(cfg);

    n_val   = round(cfg.val_fraction * cfg.n_patients);
    n_train = cfg.n_patients - n_val;
    idx     = randperm(cfg.n_patients);
    train_idx = idx(1:n_train);
    val_idx   = idx(n_train+1:end);

    X_train = dataset.X(:, :, train_idx);
    A_train = dataset.A(:, :, train_idx);
    X_val   = dataset.X(:, :, val_idx);
    A_val   = dataset.A(:, :, val_idx);
    Y_val   = dataset.Y(:, :, val_idx);

    %% Step 2: Build SINDy library and identify model
    [Theta, dX, lib_labels, protected_idx] = build_sindy_library(X_train, A_train, cfg);

    lambda_opt = cross_validate_lambda(Theta, dX, cfg, protected_idx);
    cfg.lambda = lambda_opt;

    Xi = stlsq(Theta, dX, cfg.lambda, cfg.max_iter, cfg.tol, false, ...
               cfg.max_terms, protected_idx, cfg.min_terms);

    [Xi, ~] = prune_sindy_bic(Xi, Theta, dX, lib_labels, ...
                               dataset.state_names, false, protected_idx);

    static_states = 3:dataset.n_states;
    Xi(:, static_states) = 0;

    [nmae, ~, ~] = validate_sindy_model(Xi, X_val, A_val, Y_val, lib_labels, cfg);
    fprintf('  SINDy NMAE: %.4f\n', nmae);

    %% Step 3: Set up iLQG for this persona
    clear global SINDY_MODEL;
    global SINDY_MODEL;

    % Persona-adjusted ceiling for this patient
    x0 = x0_true_base;
    swat = 0.4*x0(3) + 0.2*(1-x0(4)) + 0.2*x0(6) + 0.1*(1-x0(5));
    swat = swat * (1 - 0.5 * persona.comorbidity);
    swat = min(max(swat, 0.1), 1.0);

    % Identify top uncertain parameters (for curiosity / dual control)
    np_max = 2;
    [term_idx, state_idx] = find(Xi(:, 1:2) ~= 0);
    n_uncertain = min(np_max, length(term_idx));
    xi_idx = [term_idx(1:n_uncertain), state_idx(1:n_uncertain)];
    np = n_uncertain;

    SINDY_MODEL.Xi           = Xi;
    SINDY_MODEL.lib_cfg      = cfg;
    SINDY_MODEL.swat_ceiling = swat;
    SINDY_MODEL.xi_idx       = xi_idx;
    SINDY_MODEL.F_dyn        = 1e-3;
    SINDY_MODEL.F_param      = 1e-4;
    SINDY_MODEL.persona      = persona;

    %% Step 4: Run iLQG session loop (13 control signals)
    nx = 6; nu = n_u; ny = 2; nv = ny; nw = nx + np;
    T = 12; N = T;
    horizon_length = 6;
    u_lims = repmat([0, 1], nu, 1);     % 13 x 2
    u_lim_method = 2;

    xi_true = zeros(np, 1);
    for k = 1:np
        xi_true(k) = Xi(xi_idx(k,1), xi_idx(k,2));
    end

    xa_true = [x0; xi_true];
    c_vec   = [swat; 1e-3; 1e-4];

    x_hat = x0;
    p_hat = xi_true;
    cov_xa = blkdiag(0.01*eye(nx), diag((0.8*abs(xi_true)+0.05).^2));

    u_all  = zeros(nu, N);
    xa_all = zeros(nx+np, N+1);
    xa_all(:,1) = xa_true;

    l_gain = 0.5 * ones(nu, horizon_length);
    L_gain = zeros(nu, nx+np, horizon_length);
    u_bar  = 0.5 * ones(nu, horizon_length);

    for s = 1:N
        T_rem = min(horizon_length, N - s + 1);
        try
            [~, unew, l_gain_s, L_gain_s, ~, ~, ~, ~, ~, ~] = ...
                iLQG_function(T_rem, 1, x_hat, l_gain(:,1:T_rem), ...
                              L_gain(:,:,1:T_rem), u_bar(:,1:T_rem), ...
                              1, 1, c_vec, p_hat, cov_xa, ...
                              1, 3, u_lims, ny, nv, nw, 50, false, ...
                              0, 0, [], u_lim_method);
            % Squash from pre-tanh space to physical [0,1]
            u_raw  = unew(:,1);
            u_phys = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u_raw) + ...
                     (u_lims(:,2) + u_lims(:,1))/2;
            u_all(:,s) = u_phys;
            l_gain(:,1:T_rem) = l_gain_s;
        catch
            u_all(:,s) = 0.5 * ones(nu,1);
        end

        if s < N
            sqrtR = 1e-2 * eye(nv);
            sqrtQ = blkdiag(1e-2*eye(nx), 1e-3*eye(np));
            [~, xa_sim] = simulate_system(1, xa_all(:,s), u_all(:,s), ...
                             c_vec, sqrtR, sqrtQ, 1, 0, u_lims, u_lim_method);
            xa_all(:,s+1) = xa_sim(:,end);
            x_hat = xa_all(1:nx, s+1);
        end

        if T_rem > 1
            % Warm start in physical space to avoid feeding back saturated values
            u_bar_raw  = [unew(:,2:end), unew(:,end)];
            u_bar_phys = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u_bar_raw) + ...
                         (u_lims(:,2) + u_lims(:,1))/2;
            u_bar = u_bar_phys;
            if size(u_bar,2) < horizon_length
                u_bar = [u_bar, repmat(u_bar(:,end), 1, horizon_length-size(u_bar,2))];
            end
        end
    end

    %% Step 5: Record results
    final_scim = xa_all(1, N);
    final_bbs  = xa_all(2, N);
    init_score  = (x0(1) + x0(2)) / 2;
    final_score = (final_scim + final_bbs) / 2;
    ceiling_gap = max(swat - init_score, 0.01);
    MFG = (final_score - init_score) / ceiling_gap;

    mean_dose = mean(u_all, 2)';   % 1 x 13 mean recommended dose per therapy

    results(pidx).persona_name = persona.name;
    results(pidx).MFG          = MFG;
    results(pidx).final_scim   = final_scim;
    results(pidx).final_bbs    = final_bbs;
    results(pidx).swat         = swat;
    results(pidx).mean_dose    = mean_dose;
    results(pidx).u_all        = u_all;
    results(pidx).xa_all       = xa_all;

    % Top-3 recommended therapies
    [sorted_dose, order] = sort(mean_dose, 'descend');
    top_codes = TH.codes(order(1:3));
    results(pidx).top_codes = top_codes;
    results(pidx).top_order = order;

    fprintf('  MFG=%.3f  SCIM=%.3f  BBS=%.3f  SWAT=%.3f\n', ...
            MFG, final_scim, final_bbs, swat);
    fprintf('  Top therapies: %s (%.2f), %s (%.2f), %s (%.2f)\n', ...
            top_codes{1}, sorted_dose(1), top_codes{2}, sorted_dose(2), ...
            top_codes{3}, sorted_dose(3));
end

%% -- Summary table --------------------------------------------------------

fprintf('\n\n==================================================================\n');
fprintf('  PERSONA THERAPY RECOMMENDATION SUMMARY\n');
fprintf('==================================================================\n');
fprintf('%-22s %6s %7s %7s   %s\n', 'Persona','MFG','SCIM','BBS','Top 3 therapies (mean dose)');
fprintf('%s\n', repmat('-',1,86));
for pidx = 1:n_personas
    r = results(pidx);
    md = r.mean_dose(r.top_order(1:3));
    fprintf('%-22s %6.3f %7.3f %7.3f   %s(%.2f) %s(%.2f) %s(%.2f)\n', ...
        r.persona_name, r.MFG, r.final_scim, r.final_bbs, ...
        r.top_codes{1}, md(1), r.top_codes{2}, md(2), r.top_codes{3}, md(3));
end
fprintf('==================================================================\n\n');

% Full per-therapy dose matrix (personas x 13)
fprintf('Mean recommended dose by therapy (rows: persona, cols: U1..U13)\n');
fprintf('%-22s', 'Persona');
for j = 1:n_u, fprintf('%6s', TH.codes{j}); end
fprintf('\n%s\n', repmat('-', 1, 22 + 6*n_u));
for pidx = 1:n_personas
    fprintf('%-22s', results(pidx).persona_name);
    for j = 1:n_u, fprintf('%6.2f', results(pidx).mean_dose(j)); end
    fprintf('\n');
end
fprintf('\nTherapy key:\n');
for j = 1:n_u
    fprintf('  %-4s %s\n', TH.codes{j}, TH.names{j});
end

%% -- Plots ----------------------------------------------------------------

names_short = cellfun(@(r) r.persona_name, num2cell(results), 'UniformOutput', false);
dose_mat = cell2mat(arrayfun(@(r) r.mean_dose, results, 'UniformOutput', false)'); % personas x 13

figure('Name','Persona Comparison','Position',[40,40,1500,520]);

% (1) MFG bar chart
subplot(1,3,1);
bar([results.MFG], 'FaceColor', [0.2 0.5 0.8]);
set(gca, 'XTickLabel', names_short, 'XTick', 1:n_personas);
xtickangle(30);
ylabel('MFG (Functional Gain Ratio)');
title('Mean Functional Gain by Persona');
grid on;

% (2) Therapy recommendation heatmap (personas x 13 therapies)
subplot(1,3,2);
imagesc(dose_mat, [0 1]);
colormap(parula); colorbar;
set(gca, 'YTick', 1:n_personas, 'YTickLabel', names_short);
set(gca, 'XTick', 1:n_u, 'XTickLabel', TH.codes);
xtickangle(60);
xlabel('WHO therapy'); ylabel('Persona');
title('Persona-Specific Recommended Therapy Mix');

% (3) SCIM trajectory comparison
subplot(1,3,3);
cmap = lines(n_personas);
hold on;
for pidx = 1:n_personas
    r = results(pidx);
    plot(1:size(r.xa_all,2), r.xa_all(1,:), '-o', ...
         'Color', cmap(pidx,:), 'DisplayName', r.persona_name, 'LineWidth', 1.5);
end
xlabel('Session'); ylabel('SCIM (normalised)');
title('SCIM Trajectory by Persona');
legend('Location','southeast'); grid on;

sgtitle('Persona-Driven SCI Therapy Planning - 13 WHO Control Signals', 'FontWeight', 'bold');

%% -- Save -----------------------------------------------------------------

out_dir = 'results';
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
save(fullfile(out_dir, 'persona_comparison.mat'), 'results', 'persona_names', 'TH');
fprintf('\nResults saved to %s/persona_comparison.mat\n', out_dir);
