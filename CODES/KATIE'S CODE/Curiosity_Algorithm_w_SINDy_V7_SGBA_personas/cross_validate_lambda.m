function lambda_opt = cross_validate_lambda(Theta, dX, cfg, protected_idx)
%CROSS_VALIDATE_LAMBDA  Select the optimal STLSQ sparsity threshold lambda
%  via k-fold cross-validation on the training data.
%
%  Key design decisions:
%   - Lambda grid spans 0.001–0.35. Upper bound is lower than 0.5 to
%     prevent selection from drifting to the grid ceiling, which previously
%     caused all action terms to be zeroed out.
%   - Protected terms are passed through to STLSQ so CV folds evaluate
%     models that will always retain clinically mandated action terms.
%   - Complexity penalty breaks ties in favour of sparser models.
%   - 1-SE rule picks the largest (sparsest) lambda within 1 SE of minimum.
%
%  INPUTS
%    Theta         [N x n_terms]   library matrix (physical units)
%    dX            [N x n_states]  derivative matrix
%    cfg           configuration struct (.n_lambda, .max_iter, .tol, etc.)
%    protected_idx [1 x n_terms] logical  (optional) protected term flags
%
%  OUTPUT
%    lambda_opt  scalar  optimal sparsity threshold

if nargin < 4 || isempty(protected_idx)
    protected_idx = false(1, size(Theta, 2));
end

n_folds   = 5;
n_lambda  = cfg.n_lambda;

% Grid from 0.001 to 0.35 — upper bound deliberately below 0.5 to avoid
% grid-ceiling selection bias that strips all action terms.
lambda_raw = logspace(-3, log10(0.35), n_lambda);

% Compute a data-adaptive lower bound: lambda must be at least 10% of the
% median absolute normalised coefficient from an initial OLS fit.
% This prevents the grid from starting so low that all 55 terms survive
% (which makes the complexity penalty flat and useless).
col_std_cv = std(Theta, 0, 1);
col_std_cv(col_std_cv < 1e-10) = 1;
Theta_n_cv = Theta ./ col_std_cv;
dX_std_cv  = std(dX, 0, 1);
dX_std_cv(dX_std_cv < 1e-10) = 1;
dX_n_cv    = dX ./ dX_std_cv;
ridge_cv   = 1e-6 * eye(size(Theta_n_cv, 2));
xi_ols     = (Theta_n_cv' * Theta_n_cv + ridge_cv) \ (Theta_n_cv' * dX_n_cv);
lambda_floor = 0.10 * median(abs(xi_ols(xi_ols ~= 0)));
lambda_floor = max(lambda_floor, 0.005);   % absolute minimum of 0.005

lambda_grid = lambda_raw(lambda_raw >= lambda_floor);
if isempty(lambda_grid)
    lambda_grid = lambda_raw;   % fallback: use full grid
end

mt  = getfield_safe(cfg, 'max_terms', Inf);
mnt = getfield_safe(cfg, 'min_terms', 1);

N        = size(Theta, 1);
n_grid   = length(lambda_grid);   % actual grid size after floor trimming
cv_error = zeros(n_grid, n_folds);
cv_nnz   = zeros(n_grid, n_folds);
DEGENERATE_PENALTY = 10;

fold_idx = crossvalind('Kfold', N, n_folds);

for li = 1:n_grid
    lam = lambda_grid(li);

    for k = 1:n_folds
        test_mask  = (fold_idx == k);
        train_mask = ~test_mask;

        Theta_tr = Theta(train_mask, :);
        dX_tr    = dX(train_mask, :);
        Theta_te = Theta(test_mask, :);
        dX_te    = dX(test_mask, :);

        Xi_k = stlsq(Theta_tr, dX_tr, lam, cfg.max_iter, cfg.tol, false, ...
                     mt, protected_idx, mnt);

        % Check for degenerate: no unprotected terms survived
        unprotected_nnz = nnz(Xi_k(~protected_idx, :));
        if unprotected_nnz == 0
            cv_error(li, k) = DEGENERATE_PENALTY;
            cv_nnz(li, k)   = nnz(Xi_k);
            continue;
        end

        dX_pred  = Theta_te * Xi_k;

        nmae_per_state = zeros(1, size(dX_te, 2));
        for s = 1:size(dX_te, 2)
            r_s   = dX_te(:,s) - dX_pred(:,s);
            rng_s = max(dX_te(:,s)) - min(dX_te(:,s));
            if rng_s < 1e-10; rng_s = 1; end
            nmae_per_state(s) = mean(abs(r_s)) / rng_s;
        end
        cv_error(li, k) = mean(nmae_per_state);
        cv_nnz(li, k)   = nnz(Xi_k);
    end
end

mean_cv  = mean(cv_error, 2);   % [n_lambda x 1]
std_cv   = std(cv_error,  0, 2) / sqrt(n_folds);

% Add a mild complexity penalty to break ties in favour of sparser models.
% Penalty = alpha * (mean_nnz / max_nnz), where alpha scales relative to
% the typical CV error magnitude.  This nudges selection toward the elbow
% of the NMAE-vs-nnz curve without dominating the error signal.
mean_nnz     = mean(cv_nnz, 2);
max_nnz      = max(mean_nnz) + eps;

% Compute alpha only from valid (non-degenerate) lambda values
valid_mean_cv = mean_cv(mean_cv < DEGENERATE_PENALTY);
if isempty(valid_mean_cv)
    error('SINDy:CV', ['All lambda values produced degenerate models. ' ...
          'Check that protected_terms are correct and the data has sufficient variation.']);
end
% Complexity penalty: 3% of mean valid error, scaled by nnz fraction.
% Small enough that it only breaks near-ties; does not dominate the error.
alpha_penalty = 0.03 * mean(valid_mean_cv);
penalised_cv  = mean_cv + alpha_penalty * (mean_nnz / max_nnz);

% Find lambda that minimises penalised CV error (ignoring degenerate models)
valid_mask = mean_cv < DEGENERATE_PENALTY;
penalised_valid = penalised_cv;
penalised_valid(~valid_mask) = Inf;
[min_err_p, min_idx_p] = min(penalised_valid);
threshold_1se = min_err_p + std_cv(min_idx_p);

% 1-SE rule: among lambdas within 1 SE of the minimum-error lambda,
% pick the SMALLEST (min-error end). We rely on the complexity penalty
% above to express the sparsity preference, not the 1-SE direction.
% (Picking candidates(end) caused selection to drift to the grid ceiling.)
candidates = find(penalised_valid <= threshold_1se);
lambda_opt = lambda_grid(candidates(1));   % smallest lambda in acceptable band

% Safety: if lambda_opt would produce a degenerate model, step forward to
% the minimum-penalised-error lambda
Xi_test = stlsq(Theta, dX, lambda_opt, cfg.max_iter, cfg.tol, false, ...
                mt, protected_idx, mnt);
if nnz(Xi_test(~protected_idx, :)) == 0
    lambda_opt = lambda_grid(min_idx_p);
    if getfield_safe(cfg, 'verbose', false)
        fprintf('  [CV] Selected lambda produced degenerate model; using min-error lambda = %.4f\n', lambda_opt);
    end
end

%% ── Plot CV curve ────────────────────────────────────────────────────────

figure('Name', 'Lambda Cross-Validation', 'NumberTitle', 'off', ...
       'Position', [100, 100, 750, 450]);

% Left axis: CV error
yyaxis left;
semilogx(lambda_grid, mean_cv, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
hold on;
fill([lambda_grid, fliplr(lambda_grid)], ...
     [mean_cv + std_cv; flipud(mean_cv - std_cv)]', ...
     [0.2, 0.5, 0.8], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
xline(lambda_opt, 'r--', sprintf('\\lambda^* = %.4f', lambda_opt), ...
      'LineWidth', 1.8, 'LabelVerticalAlignment', 'bottom', 'FontSize', 10);
ylabel('Mean CV NMAE', 'FontSize', 11);
valid_cv = mean_cv(mean_cv < DEGENERATE_PENALTY);
if isempty(valid_cv)
    ylim([0, DEGENERATE_PENALTY]);
else
    ylim([0, min(DEGENERATE_PENALTY * 0.8, max(valid_cv) * 2)]);
end

% Right axis: mean nnz
yyaxis right;
mean_nnz = mean(cv_nnz, 2);
semilogx(lambda_grid, mean_nnz, 'g--^', 'LineWidth', 1.2, 'MarkerSize', 5);
ylabel('Mean non-zero coefficients', 'FontSize', 11);

xlabel('\lambda (sparsity threshold, normalised space)', 'FontSize', 11);
title('SINDy Cross-Validation: Sparsity Threshold Selection', 'FontSize', 12);
legend({'CV NMAE (mean)', 'NMAE ± SE', 'Optimal \lambda', 'Non-zero terms'}, ...
       'Location', 'northwest', 'FontSize', 9);
grid on;
set(gca, 'FontSize', 10);
drawnow;

end

%% ── Helper ───────────────────────────────────────────────────────────────

function v = getfield_safe(s, field, default)
%GETFIELD_SAFE  Return s.field if it exists, otherwise return default.
if isfield(s, field)
    v = s.(field);
else
    v = default;
end
end
