function plot_results(sim_traj, Y_val, dataset, cfg)
%PLOT_RESULTS  Visualise SINDy model validation results.
%
%  Produces four figures:
%    Fig 1 - Sample trajectory comparisons (actual vs. simulated)
%    Fig 2 - Prediction error distribution across validation patients
%    Fig 3 - Mean functional gain: oracle vs. clinical baseline placeholders
%    Fig 4 - SINDy coefficient heatmap (sparsity pattern)
%
%  INPUTS
%    sim_traj  struct   output of validate_sindy_model
%    Y_val     array    validation outcomes
%    dataset   struct   dataset metadata
%    cfg       struct   configuration

sessions = 1:sim_traj.n_sessions;
n_val    = sim_traj.n_val;
state_idx = sim_traj.state_idx;   % [1, 2] = SCIM, BBS

state_labels = dataset.state_names(state_idx);
colors = lines(min(n_val, 6));

%% ── Figure 1: Sample trajectory comparisons ─────────────────────────────

n_show = min(6, n_val);
fig1 = figure('Name', 'Trajectory Comparisons', ...
              'NumberTitle', 'off', ...
              'Position', [100, 100, 1200, 500]);

for si = 1:length(state_idx)
    subplot(1, length(state_idx), si);
    hold on;

    for p = 1:n_show
        s_idx = state_idx(si);
        h1 = plot(sessions, squeeze(sim_traj.X_true(s_idx, :, p)), ...
             '-o', 'Color', colors(p,:), 'LineWidth', 1.5, ...
             'MarkerSize', 5, 'DisplayName', sprintf('True P%d', p));

        h2 = plot(sessions, squeeze(sim_traj.X_sim(s_idx, :, p)), ...
             '--s', 'Color', colors(p,:), 'LineWidth', 1.2, ...
             'MarkerSize', 4, 'MarkerFaceColor', 'w', ...
             'DisplayName', sprintf('SINDy P%d', p));
    end

    xlabel('Session (weeks)', 'FontSize', 11);
    ylabel(sprintf('%s (normalised)', state_labels{si}), 'FontSize', 11);
    title(sprintf('Actual vs. SINDy-simulated: %s', state_labels{si}), ...
          'FontSize', 12);
    grid on;
    set(gca, 'FontSize', 10);

    % Add legend entry for solid/dashed line types only
    h_leg = [plot(NaN, NaN, 'k-o', 'LineWidth', 1.5), ...
             plot(NaN, NaN, 'k--s', 'LineWidth', 1.2)];
    legend(h_leg, {'Actual', 'SINDy'}, 'Location', 'northwest', ...
           'FontSize', 9);
end

sgtitle(sprintf('SINDy Forward Simulation Validation  (N_{val} = %d)', n_val), ...
        'FontSize', 13, 'FontWeight', 'bold');

%% ── Figure 2: Error distribution ────────────────────────────────────────

fig2 = figure('Name', 'Prediction Error Distribution', ...
              'NumberTitle', 'off', 'Position', [150, 150, 700, 450]);

all_errors = [];
for p = 1:n_val
    for si = 1:length(state_idx)
        s_idx = state_idx(si);
        err = squeeze(sim_traj.X_true(s_idx, :, p)) - ...
              squeeze(sim_traj.X_sim(s_idx, :, p));
        all_errors = [all_errors, err]; %#ok<AGROW>
    end
end

histogram(all_errors, 40, 'FaceColor', [0.2, 0.5, 0.8], 'EdgeColor', 'w');
hold on;
xline(mean(all_errors), 'r--', sprintf('Mean = %.4f', mean(all_errors)), ...
      'LineWidth', 2, 'LabelVerticalAlignment', 'top', 'FontSize', 10);
xline(0, 'k-', 'LineWidth', 1);

xlabel('Prediction Error (normalised)', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('SINDy Prediction Error Distribution (Validation Set)', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 10);

dyn_range = max(sim_traj.X_true(state_idx,:,:), [], 'all') - ...
            min(sim_traj.X_true(state_idx,:,:), [], 'all');
nmae_plot = mean(abs(all_errors)) / dyn_range;
annotation('textbox', [0.62, 0.75, 0.3, 0.12], ...
    'String', sprintf('NMAE = %.4f\nRMSE = %.4f', ...
                      nmae_plot, sqrt(mean(all_errors.^2))), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w', ...
    'EdgeColor', [0.5,0.5,0.5], 'FontSize', 10);

%% ── Figure 3: Functional gain comparison (placeholder structure) ─────────

fig3 = figure('Name', 'Functional Gain Comparison', ...
              'NumberTitle', 'off', 'Position', [200, 200, 650, 500]);

% Compute per-patient total functional gain from simulation
sindy_gains = zeros(1, n_val);
for p = 1:n_val
    scim_gain = sim_traj.X_sim(1, end, p) - sim_traj.X_sim(1, 1, p);
    bbs_gain  = sim_traj.X_sim(2, end, p) - sim_traj.X_sim(2, 1, p);
    sindy_gains(p) = (scim_gain + bbs_gain) / 2;
end

% Oracle and baseline placeholders (replace with real values)
oracle_gains   = sindy_gains * 1.15 + 0.02*randn(1,n_val);  % PLACEHOLDER
baseline_gains = sindy_gains * 0.75 + 0.02*randn(1,n_val);  % PLACEHOLDER

all_gains = {baseline_gains, sindy_gains, oracle_gains};
labels_bp = {'Clinical Baseline', 'SINDy (curiosity-driven)', 'Oracle'};
bp_colors = [0.7 0.7 0.7; 0.2 0.5 0.8; 0.9 0.4 0.2];

hold on;
for g = 1:3
    b = boxplot(all_gains{g}, 'Positions', g, ...
                'Widths', 0.5, 'Symbol', 'o');
    set(b, 'LineWidth', 1.5);
    patch([g-0.25, g+0.25, g+0.25, g-0.25], ...
          [prctile(all_gains{g},25), prctile(all_gains{g},25), ...
           prctile(all_gains{g},75), prctile(all_gains{g},75)], ...
          bp_colors(g,:), 'FaceAlpha', 0.4, 'EdgeColor', 'none');
end

set(gca, 'XTick', 1:3, 'XTickLabel', labels_bp, 'FontSize', 10);
ylabel('Mean Functional Gain (normalised)', 'FontSize', 11);
title({'Functional Gain: SINDy vs. Baselines'; ...
       '{\color{red}[PLACEHOLDER — replace with real results]}'}, ...
      'FontSize', 12);
grid on; box on;
annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
    'String', 'Note: Oracle and Clinical Baseline values are illustrative placeholders.', ...
    'HorizontalAlignment', 'center', 'FontSize', 8, ...
    'EdgeColor', 'none', 'Color', [0.5 0 0]);

%% ── Figure 4: SINDy coefficient heatmap ─────────────────────────────────
% (loaded from saved model if available)

if isfield(cfg, 'output_dir') && exist(fullfile(cfg.output_dir, 'sindy_model.mat'), 'file')
    S = load(fullfile(cfg.output_dir, 'sindy_model.mat'), 'Xi', 'lib_labels');
    fig4 = figure('Name', 'SINDy Coefficient Heatmap', ...
                  'NumberTitle', 'off', 'Position', [250, 250, 900, 500]);

    % Show only dynamic states
    Xi_show = S.Xi(:, 1:2);
    n_terms_show = min(30, size(Xi_show,1));  % truncate for readability

    imagesc(Xi_show(1:n_terms_show, :));
    colormap(redblue_colormap());
    colorbar;
    clim_val = max(abs(Xi_show(:)));
    if clim_val > 0
        clim([-clim_val, clim_val]);
    end

    % Strip scale tags for y-labels
    raw_lbl = regexprep(S.lib_labels(1:n_terms_show), '\s*\[s=[\d.]+\]', '');
    yticks(1:n_terms_show);
    yticklabels(raw_lbl);
    xticks(1:2);
    xticklabels({'SCIM dynamics', 'BBS dynamics'});
    set(gca, 'FontSize', 9, 'TickLabelInterpreter', 'none');
    title('SINDy Identified Coefficients (non-zero = active term)', ...
          'FontSize', 12);
    xlabel('State equation', 'FontSize', 11);
    ylabel('Library term', 'FontSize', 11);
end

%% ── Save figures ─────────────────────────────────────────────────────────

if cfg.save_results
    saveas(fig1, fullfile(cfg.output_dir, 'fig1_trajectories.png'));
    saveas(fig2, fullfile(cfg.output_dir, 'fig2_error_dist.png'));
    saveas(fig3, fullfile(cfg.output_dir, 'fig3_functional_gain.png'));
    fprintf('Figures saved to %s/\n', cfg.output_dir);
end

end

%% ── Red-blue diverging colormap ──────────────────────────────────────────

function cmap = redblue_colormap()
r = linspace(0.8, 1, 64)';   g_up = linspace(0, 1, 64)';   b_up = linspace(0, 1, 64)';
r2 = linspace(1, 0.2, 64)';  g_dn = linspace(1, 0, 64)';   b_dn = ones(64,1);
cmap = [b_up, g_up, r; b_dn, g_dn, r2];
% Blue (negative) to white to red (positive)
cmap = flipud([linspace(0.2,1,128)', linspace(0,1,128)', linspace(1,0.2,128)']);
cmap = [linspace(0.2,1,64)', linspace(0,1,64)', ones(64,1);
        ones(64,1),         linspace(1,0,64)', linspace(1,0.2,64)'];
end
