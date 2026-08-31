%% =========================================================================
%  compare_two_personas.m
%  Compare the recommended therapy mix for TWO personas side by side.
%
%  Defaults to "Pain Sensitive" vs "High Motivator", but you can change the
%  two names below to compare any pair. For each persona it:
%    1. Generates a persona-specific synthetic dataset (13 WHO therapies)
%    2. Identifies the persona's dynamics with SINDy
%    3. Runs the dual (curiosity-driven) iLQG planner with that persona's
%       cost weights and the uniform resource-preservation penalty
%    4. Records the mean recommended dose of each of the 13 therapies
%  then draws a grouped bar chart so the two plans can be read at a glance.
%
%  Author : Katie Campbell, UNB ECE
%  Date   : June 2026
% =========================================================================

clear; clc; close all;
clear global SINDY_MODEL;
rng(42);

%% ── Choose the two personas to compare ───────────────────────────────────
PERSONA_A = 'pain_sensitive';
PERSONA_B = 'high_motivator';
pair = {PERSONA_A, PERSONA_B};

TH  = who_therapies();
n_u = TH.n;                       % 13

%% ── Shared config (kept modest so the script runs quickly) ───────────────
base_cfg = struct();
base_cfg.n_patients    = 120;
base_cfg.n_sessions    = 16;
base_cfg.noise_std     = 0.02;
base_cfg.poly_order    = 2;
base_cfg.include_trig  = false;
base_cfg.include_cross = true;
base_cfg.lambda        = 0.01;
base_cfg.max_iter      = 100;
base_cfg.tol           = 1e-6;
base_cfg.nmae_threshold= 1.0;
base_cfg.max_terms     = 22;
base_cfg.min_terms     = 3;
base_cfg.protected_terms = {'x3*a3', 'x6*a8'};
base_cfg.val_fraction  = 0.2;
base_cfg.n_lambda      = 20;
base_cfg.verbose       = false;
base_cfg.save_results  = false;

% Same patient presented to each persona-tuned planner (fair comparison).
x0_true = [0.25; 0.20; 0.75; 0.45; 0.30; 0.70];   % SCIM BBS AIS Age DPI CG

dose = zeros(2, n_u);             % rows: personas, cols: U1..U13
labels = cell(1, 2);

for pidx = 1:2
    persona = define_persona(pair{pidx});
    labels{pidx} = persona.name;
    fprintf('\n=== %s ===\n', persona.name);

    %% 1. Persona-specific dataset
    cfg = base_cfg; cfg.persona = persona;
    dataset = generate_synthetic_dataset(cfg);

    %% 2. SINDy identification
    [Theta, dX, lib_labels, protected_idx] = build_sindy_library(dataset.X, dataset.A, cfg);
    Xi = stlsq(Theta, dX, cfg.lambda, cfg.max_iter, cfg.tol, false, ...
               cfg.max_terms, protected_idx, cfg.min_terms);
    Xi(:, 3:dataset.n_states) = 0;     % static states have no dynamics

    %% 3. iLQG planner setup (persona-specific model + weights)
    clear global SINDY_MODEL; global SINDY_MODEL;
    swat = 0.4*x0_true(3) + 0.2*(1-x0_true(4)) + 0.2*x0_true(6) + 0.1*(1-x0_true(5));
    swat = swat * (1 - 0.5 * persona.comorbidity);
    swat = min(max(swat, 0.1), 1.0);

    np_max = 2;
    [term_idx, state_idx] = find(Xi(:, 1:2) ~= 0);
    np = min(np_max, numel(term_idx));
    xi_idx = [term_idx(1:np), state_idx(1:np)];

    SINDY_MODEL.Xi = Xi; SINDY_MODEL.lib_cfg = cfg; SINDY_MODEL.swat_ceiling = swat;
    SINDY_MODEL.xi_idx = xi_idx; SINDY_MODEL.F_dyn = 1e-3; SINDY_MODEL.F_param = 1e-4;
    SINDY_MODEL.persona = persona;

    nx = 6; nu = n_u; ny = 2; nv = ny; nw = nx + np;
    N = 12; horizon = 6;
    u_lims = repmat([0, 1], nu, 1); u_lim_method = 2;

    xi_true = zeros(np,1);
    for k = 1:np; xi_true(k) = Xi(xi_idx(k,1), xi_idx(k,2)); end
    c_vec = [swat; 1e-3; 1e-4];

    x_hat = x0_true; p_hat = xi_true;
    cov_xa = blkdiag(0.01*eye(nx), diag((0.8*abs(xi_true)+0.05).^2));
    xa_all = [x0_true; xi_true];
    u_all = zeros(nu, N);
    l_gain = 0.5*ones(nu, horizon); L_gain = zeros(nu, nx+np, horizon);
    u_bar  = 0.5*ones(nu, horizon);

    %% 4. Session loop
    for s = 1:N
        T_rem = min(horizon, N - s + 1);
        try
            [~, unew, l_s] = iLQG_function(T_rem, 1, x_hat, l_gain(:,1:T_rem), ...
                L_gain(:,:,1:T_rem), u_bar(:,1:T_rem), 1, 1, c_vec, p_hat, cov_xa, ...
                1, 3, u_lims, ny, nv, nw, 50, false, 0, 0, [], u_lim_method);
            u_phys = (u_lims(:,2)-u_lims(:,1))/2 .* tanh(unew(:,1)) + (u_lims(:,2)+u_lims(:,1))/2;
            u_all(:,s) = u_phys;
            l_gain(:,1:T_rem) = l_s;
        catch
            u_all(:,s) = 0.5*ones(nu,1);
        end
        if s < N
            [~, xa_sim] = simulate_system(1, xa_all, u_all(:,s), c_vec, ...
                1e-2*eye(nv), blkdiag(1e-2*eye(nx),1e-3*eye(np)), 1, 0, u_lims, u_lim_method);
            xa_all = xa_sim(:,end); x_hat = xa_all(1:nx);
        end
        if T_rem > 1
            ub_raw = [unew(:,2:end), unew(:,end)];
            u_bar = (u_lims(:,2)-u_lims(:,1))/2 .* tanh(ub_raw) + (u_lims(:,2)+u_lims(:,1))/2;
            if size(u_bar,2) < horizon
                u_bar = [u_bar, repmat(u_bar(:,end),1,horizon-size(u_bar,2))];
            end
        end
    end

    dose(pidx,:) = mean(u_all, 2)';
    [sv, ord] = sort(dose(pidx,:), 'descend');
    fprintf('  Top therapies: %s(%.2f) %s(%.2f) %s(%.2f)\n', ...
        TH.codes{ord(1)}, sv(1), TH.codes{ord(2)}, sv(2), TH.codes{ord(3)}, sv(3));
end

%% ── Grouped bar plot ─────────────────────────────────────────────────────
figure('Name','Therapy recommendations by persona','Position',[80,80,1150,560]);
hb = bar(dose', 'grouped');
hb(1).FaceColor = [0.88 0.48 0.37];     % Pain Sensitive (coral)
hb(2).FaceColor = [0.06 0.43 0.34];     % High Motivator (teal)
set(gca, 'XTick', 1:n_u, 'XTickLabel', TH.codes);
xtickangle(45);
ylabel('Mean recommended dose  [0 = none, 1 = maximum]');
xlabel('WHO therapy');
title(sprintf('Recommended therapy mix: %s vs %s', labels{1}, labels{2}));
legend(labels, 'Location', 'northeast'); grid on; ylim([0, 1]);

% Therapy key under the axes
key = '';
for j = 1:n_u
    key = [key, sprintf('%s = %s', TH.codes{j}, TH.names{j})];
    if j < n_u; key = [key, '   ·   ']; end
end
annotation('textbox', [0.02 0.0 0.96 0.06], 'String', key, ...
    'EdgeColor','none', 'FontSize', 7, 'Interpreter','none', ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom');

%% ── Save ─────────────────────────────────────────────────────────────────
if ~exist('results','dir'); mkdir('results'); end
saveas(gcf, fullfile('results', 'persona_pair_comparison.png'));
fprintf('\nSaved plot to results/persona_pair_comparison.png\n');
