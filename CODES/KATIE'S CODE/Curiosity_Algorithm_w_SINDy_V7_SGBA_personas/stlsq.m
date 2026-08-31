function Xi = stlsq(Theta, dX, lambda, max_iter, tol, verbose, max_terms, protected_idx, min_terms)
%STLSQ  Sequential Thresholded Least Squares for SINDy sparse regression.
%
%  Recovers the sparse coefficient matrix Xi such that:
%    dX ≈ Theta * Xi
%  by iteratively applying least squares and thresholding small coefficients.
%
%  Protected terms (protected_idx = true) are never zeroed by thresholding,
%  ensuring action terms survive regardless of coefficient magnitude.
%  min_terms enforces a floor on active terms per state equation.
%
%  INPUTS
%    Theta         [N x n_terms]   library matrix (physical units)
%    dX            [N x n_states]  state derivatives
%    lambda        scalar          sparsity threshold (normalised space)
%    max_iter      scalar          maximum iterations
%    tol           scalar          convergence tolerance
%    verbose       logical         print iteration info
%    max_terms     scalar (opt.)   hard cap on active terms (default: Inf)
%    protected_idx [1 x n_terms] logical  terms immune to thresholding (default: all false)
%    min_terms     scalar (opt.)   minimum active terms per state (default: 1)
%
%  OUTPUT
%    Xi       [n_terms x n_states]  sparse coefficient matrix (physical units)

if nargin < 7 || isempty(max_terms);     max_terms     = Inf;                        end
if nargin < 8 || isempty(protected_idx); protected_idx = false(1, size(Theta, 2));   end
if nargin < 9 || isempty(min_terms);     min_terms     = 1;                          end

protected_idx = logical(protected_idx(:)');   % ensure row logical vector

[~, n_terms]  = size(Theta);
[~, n_states] = size(dX);

%% ── Internal normalisation ───────────────────────────────────────────────
col_std = std(Theta, 0, 1);
col_std(col_std < 1e-10) = 1;
Theta_n = Theta ./ col_std;

dX_std = std(dX, 0, 1);
dX_std(dX_std < 1e-10) = 1;
dX_n   = dX ./ dX_std;

%% ── Initialise with ridge-regularised least-squares ─────────────────────
ridge = 1e-6 * eye(n_terms);
Xi_n  = (Theta_n' * Theta_n + ridge) \ (Theta_n' * dX_n);

if verbose
    fprintf('  STLSQ: initial nnz = %d / %d  (lambda = %.4f)\n', ...
            nnz(abs(Xi_n) >= lambda), numel(Xi_n), lambda);
end

%% ── STLSQ iterations ─────────────────────────────────────────────────────
for iter = 1:max_iter
    Xi_prev = Xi_n;

    % Threshold: zero out normalised coefficients below lambda,
    % but NEVER zero protected terms
    small = (abs(Xi_n) < lambda) & ~repmat(protected_idx', 1, n_states);
    Xi_n(small) = 0;

    for s = 1:n_states
        active = find(~small(:, s) | (Xi_n(:, s) ~= 0));

        % Enforce minimum terms: always keep protected + highest-magnitude
        % unprotected terms to reach min_terms floor
        if sum(Xi_n(:,s) ~= 0) < min_terms
            % Find best non-active candidates by initial |coefficient|
            inactive     = find(Xi_n(:,s) == 0 & ~protected_idx');
            [~, sort_in] = sort(abs(Xi_prev(inactive, s)), 'descend');
            needed       = min_terms - sum(Xi_n(:,s) ~= 0);
            revive       = inactive(sort_in(1:min(needed, length(sort_in))));
            Xi_n(revive, s) = Xi_prev(revive, s);   % restore previous value
            active = find(Xi_n(:, s) ~= 0);
        end

        if isempty(active)
            Xi_n(:, s) = 0;
            continue;
        end

        Theta_active = Theta_n(:, active);
        ridge_act    = 1e-8 * eye(length(active));
        xi_act       = (Theta_active' * Theta_active + ridge_act) \ ...
                       (Theta_active' * dX_n(:, s));
        Xi_n(:, s)      = 0;
        Xi_n(active, s) = xi_act;
    end

    delta = norm(Xi_n - Xi_prev, 'fro') / (norm(Xi_prev, 'fro') + eps);
    if verbose && mod(iter, 10) == 0
        fprintf('  Iter %3d | delta = %.2e | nnz = %d\n', iter, delta, nnz(Xi_n));
    end
    if delta < tol
        if verbose
            fprintf('  Converged at iter %d (delta = %.2e)\n', iter, delta);
        end
        break;
    end
end

%% ── Hard cap: prune to max_terms per state equation ─────────────────────
% Never prune protected terms during the cap pass.
if isfinite(max_terms)
    for s = 1:n_states
        active = find(Xi_n(:, s) ~= 0);
        while length(active) > max_terms
            % Among active non-protected terms, remove smallest magnitude
            active_unprotected = active(~protected_idx(active)');
            if isempty(active_unprotected); break; end

            [~, prune_local] = min(abs(Xi_n(active_unprotected, s)));
            prune_global     = active_unprotected(prune_local);
            active(active == prune_global) = [];

            if isempty(active)
                Xi_n(:, s) = 0;
                break;
            end
            Theta_active = Theta_n(:, active);
            ridge_act    = 1e-8 * eye(length(active));
            xi_act       = (Theta_active' * Theta_active + ridge_act) \ ...
                           (Theta_active' * dX_n(:, s));
            Xi_n(:, s)        = 0;
            Xi_n(active, s)   = xi_act;
        end
    end
    if verbose
        fprintf('  After max_terms cap: nnz = %d / %d\n', nnz(Xi_n), numel(Xi_n));
    end
end

%% ── De-normalise ─────────────────────────────────────────────────────────
Xi = Xi_n .* (dX_std ./ col_std');

if verbose
    fprintf('  STLSQ: final nnz = %d / %d\n', nnz(Xi), numel(Xi));
end

end
