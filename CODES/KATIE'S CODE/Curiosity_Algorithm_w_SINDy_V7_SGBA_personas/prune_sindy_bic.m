function [Xi_pruned, bic_history] = prune_sindy_bic(Xi, Theta, dX, lib_labels, state_names, verbose, protected_idx)
%PRUNE_SINDY_BIC  Post-selection BIC pruning of a SINDy model.
%
%  After STLSQ converges, iteratively removes the lowest-magnitude active
%  term from each dynamic state equation and accepts the removal only if
%  BIC does not increase. Protected terms are never candidates for removal.
%
%  BIC = N * log(RSS/N) + k * log(N)
%
%  INPUTS
%    Xi            [n_terms x n_states]  sparse coefficients (physical units)
%    Theta         [N x n_terms]         library matrix (physical units)
%    dX            [N x n_states]        state derivatives
%    lib_labels    {1 x n_terms}         library term labels (with scale tags)
%    state_names   {1 x n_states}        state variable names
%    verbose       logical               print pruning steps
%    protected_idx [1 x n_terms] logical terms immune to pruning (default: all false)
%
%  OUTPUTS
%    Xi_pruned    [n_terms x n_states]  pruned coefficient matrix
%    bic_history  struct array          BIC and nnz trace per state equation

if nargin < 7 || isempty(protected_idx)
    protected_idx = false(1, size(Theta, 2));
end
protected_idx = logical(protected_idx(:)');   % ensure 1 x n_terms row vector

[N, n_terms]  = size(Theta);
[~, n_states] = size(dX);

Xi_pruned   = Xi;
bic_history = struct('state', {}, 'bic_vals', {}, 'nnz_vals', {});

%% ── Internal normalisation (matches stlsq convention) ────────────────────

col_std = std(Theta, 0, 1);
col_std(col_std < 1e-10) = 1;
Theta_n = Theta ./ col_std;

dX_std = std(dX, 0, 1);
dX_std(dX_std < 1e-10) = 1;
dX_n = dX ./ dX_std;

%% ── BIC pruning per dynamic state (SCIM and BBS only) ────────────────────

for s = 1:min(2, n_states)

    active = find(Xi_pruned(:, s) ~= 0);
    if isempty(active)
        continue;
    end

    % Convert physical coefficients to normalised space for BIC computation
    scale_s = dX_std(s) ./ col_std';        % [n_terms x 1] de-normalisation factor
    xi_s_n  = Xi_pruned(:, s) ./ scale_s;   % normalised coefficients

    % Initialise BIC trace
    max_steps = length(active) + 1;
    bic_trace = nan(1, max_steps);
    nnz_trace = nan(1, max_steps);
    step = 1;

    dX_pred_n = Theta_n(:, active) * xi_s_n(active);
    rss       = sum((dX_n(:, s) - dX_pred_n).^2);
    bic_curr  = N * log(rss / N + eps) + length(active) * log(N);

    bic_trace(step) = bic_curr;
    nnz_trace(step) = length(active);

    if verbose
        fprintf('\n  BIC pruning: d(%s)/dt\n', state_names{s});
        fprintf('    Initial: %d terms, BIC = %.2f\n', length(active), bic_curr);
    end

    %% Greedy backward elimination — never drop protected terms
    keep_pruning = true;
    while keep_pruning && length(active) > 1

        % Candidate terms: active AND not protected
        droppable = active(~protected_idx(active)');
        if isempty(droppable)
            if verbose
                fprintf('    Stopped (only protected terms remain)\n');
            end
            break;
        end

        % Pick the droppable term with the smallest normalised |coefficient|
        [~, drop_local] = min(abs(xi_s_n(droppable)));
        candidate_drop  = droppable(drop_local);
        trial_active    = active(active ~= candidate_drop);

        % Refit on trial active set
        Theta_tr   = Theta_n(:, trial_active);
        ridge_t    = 1e-8 * eye(length(trial_active));
        xi_trial_n = zeros(n_terms, 1);
        xi_trial_n(trial_active) = ...
            (Theta_tr' * Theta_tr + ridge_t) \ (Theta_tr' * dX_n(:, s));

        % Compute trial BIC
        rss_trial = sum((dX_n(:, s) - ...
                         Theta_n(:, trial_active) * xi_trial_n(trial_active)).^2);
        bic_trial = N * log(rss_trial / N + eps) + length(trial_active) * log(N);

        raw_lbl = regexprep(lib_labels{candidate_drop}, '\s*\[s=[\d.]+\]', '');

        if bic_trial <= bic_curr + 0.5   % accept: BIC does not meaningfully increase
            active   = trial_active;
            xi_s_n   = xi_trial_n;
            bic_curr = bic_trial;

            step = step + 1;
            bic_trace(step) = bic_curr;
            nnz_trace(step) = length(active);

            if verbose
                fprintf('    Dropped "%s" -> %d terms, BIC = %.2f\n', ...
                        raw_lbl, length(active), bic_trial);
            end
        else
            keep_pruning = false;
            if verbose
                fprintf('    Stopped (dropping "%s" would increase BIC by %.2f)\n', ...
                        raw_lbl, bic_trial - bic_curr);
            end
        end

    end   % while keep_pruning

    %% Write pruned coefficients back to physical Xi
    xi_phys = zeros(n_terms, 1);
    xi_phys(active) = xi_s_n(active) .* scale_s(active);
    Xi_pruned(:, s) = xi_phys;

    bic_history(end+1).state    = state_names{s};  %#ok<AGROW>
    bic_history(end).bic_vals   = bic_trace(1:step);
    bic_history(end).nnz_vals   = nnz_trace(1:step);

    if verbose
        fprintf('    Final: %d terms retained (from %d)\n', ...
                length(active), nnz(Xi(:, s)));
    end

end   % for s

end   % function
