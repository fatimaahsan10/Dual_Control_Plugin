function [Theta, dX, labels, protected_idx] = build_sindy_library(X, A, cfg)
%BUILD_SINDY_LIBRARY  Construct the SINDy candidate function library Theta
%  and the corresponding state-derivative matrix dX from trajectory data.
%
%  Also returns protected_idx: a logical vector marking library columns
%  that must never be zeroed by STLSQ thresholding (action terms and
%  key clinically-mandated interactions from cfg.protected_terms).
%
%  The library includes:
%    - Constant term (bias)
%    - Linear terms in state variables x1...xn and actions a1...am
%    - Polynomial terms up to cfg.poly_order
%    - Pairwise cross-products (if cfg.include_cross = true)
%    - Trigonometric terms sin/cos (if cfg.include_trig = true)
%
%  Clinically motivated cross-terms are prioritised:
%    x1*x3 (SCIM x AIS), x2*x3 (BBS x AIS), a1*x3 (intensity x AIS),
%    x6*a1 (caregiver x intensity) — consistent with clinician interviews.
%
%  INPUTS
%    X   [n_states  x n_sessions x n_patients]  state trajectories
%    A   [n_actions x n_sessions x n_patients]  actions
%    cfg  configuration struct
%
%  OUTPUTS
%    Theta  [N x n_terms]  library matrix (N = total observations)
%    dX     [N x n_states] state derivatives (finite differences)
%    labels {1 x n_terms}  human-readable term names

[n_states,  n_sessions, n_patients] = size(X);
[n_actions, ~,          ~         ] = size(A);

%% ── Compute state derivatives via finite differences ─────────────────────
%   Using second-order central differences where possible,
%   forward/backward at endpoints.

dX_full = zeros(n_states, n_sessions, n_patients);
dt = 1;  % 1-week session interval

for p = 1:n_patients
    Xp = X(:, :, p);

    % Forward difference at t=1
    dX_full(:, 1, p) = (Xp(:,2) - Xp(:,1)) / dt;

    % Central differences t=2..T-1
    for s = 2:n_sessions-1
        dX_full(:, s, p) = (Xp(:,s+1) - Xp(:,s-1)) / (2*dt);
    end

    % Backward difference at t=T
    dX_full(:, n_sessions, p) = ...
        (Xp(:,n_sessions) - Xp(:,n_sessions-1)) / dt;
end

%% ── Stack observations into matrices ──────────────────────────────────────
%   Use interior time points only (exclude endpoints to reduce noise)
%   Each row = one (patient, session) observation.

obs_sessions = 2:n_sessions-1;   % interior sessions
N = n_patients * length(obs_sessions);

Z  = zeros(N, n_states + n_actions);  % combined state+action matrix
dX = zeros(N, n_states);

row = 1;
for p = 1:n_patients
    for s = obs_sessions
        state_vec  = X(:, s, p)';    % 1 x n_states
        action_vec = A(:, s, p)';    % 1 x n_actions
        deriv_vec  = dX_full(:, s, p)';  % 1 x n_states

        Z(row, :)  = [state_vec, action_vec];
        dX(row, :) = deriv_vec;
        row = row + 1;
    end
end

%% ── Build library columns ────────────────────────────────────────────────

Theta_cols = {};
labels     = {};

% Variable names for label generation
var_names = cell(1, n_states + n_actions);
for i = 1:n_states
    var_names{i} = sprintf('x%d', i);
end
for j = 1:n_actions
    var_names{n_states + j} = sprintf('a%d', j);
end

n_vars = n_states + n_actions;

%% 1. Constant (bias) term
Theta_cols{end+1} = ones(N, 1);
labels{end+1}     = '1';

%% 2. Linear terms
for i = 1:n_vars
    Theta_cols{end+1} = Z(:, i);
    labels{end+1}     = var_names{i};
end

%% 3. Polynomial terms (degree 2..poly_order)
if cfg.poly_order >= 2
    % Squared terms
    for i = 1:n_vars
        Theta_cols{end+1} = Z(:,i).^2;
        labels{end+1}     = sprintf('%s^2', var_names{i});
    end

    % Pairwise cross-products
    if cfg.include_cross
        for i = 1:n_vars
            for j = i+1:n_vars
                Theta_cols{end+1} = Z(:,i) .* Z(:,j);
                labels{end+1}     = sprintf('%s*%s', var_names{i}, var_names{j});
            end
        end
    end
end

if cfg.poly_order >= 3
    % Cubic terms (selected interactions only to keep library tractable)
    for i = 1:min(n_states, 2)   % x1^3, x2^3 only
        Theta_cols{end+1} = Z(:,i).^3;
        labels{end+1}     = sprintf('%s^3', var_names{i});
    end
end

%% 4. Trigonometric terms (optional, not typical for rehabilitation dynamics)
if cfg.include_trig
    for i = 1:n_vars
        Theta_cols{end+1} = sin(Z(:,i));
        labels{end+1}     = sprintf('sin(%s)', var_names{i});
    end
    for i = 1:n_vars
        Theta_cols{end+1} = cos(Z(:,i));
        labels{end+1}     = sprintf('cos(%s)', var_names{i});
    end
end

%% ── Assemble Theta ───────────────────────────────────────────────────────

Theta = [Theta_cols{:}];  % [N x n_terms]

% Compute column standard deviations for two purposes:
%   (a) stored in labels so print_identified_equations can de-normalise
%   (b) STLSQ will apply normalisation internally on a per-call basis
%       so that lambda thresholding is on a consistent O(1) scale.
% NOTE: we do NOT divide Theta here. Normalisation is applied inside
%       stlsq() so the caller always works with physical-unit columns.
%       This keeps lambda interpretable across datasets of different sizes.
col_std = std(Theta, 0, 1);
col_std(col_std < 1e-10) = 1;   % constant column (bias) gets scale = 1

% Attach column scaling to labels for equation reconstruction
for k = 1:length(labels)
    labels{k} = sprintf('%s [s=%.3f]', labels{k}, col_std(k));
end

%% ── Build protected_idx ──────────────────────────────────────────────────
% Mark library columns that must survive STLSQ thresholding.
%
% DEFAULT protection: only the three pure linear action terms
%   (a1, a2, a3 — exact label match, not substring).
% These ensure the controller always has at least one direct action lever.
%
% ADDITIONAL protection: any terms in cfg.protected_terms are matched as
% exact label strings (e.g. 'x3*a1', 'x6*a1').
%
% Keeping the protected set small (3–6 terms) is important: protecting too
% many terms defeats the purpose of sparsity regularisation.

n_terms       = length(labels);
protected_idx = false(1, n_terms);

% Strip scale tags for clean matching
raw_labels_clean = regexprep(labels, '\s*\[s=[\d.]+\]', '');

% Default: exact-match pure linear action terms only
default_protect_exact = {};
for j = 1:n_actions
    default_protect_exact{end+1} = sprintf('a%d', j); %#ok<AGROW>
end

% User-specified additional terms (exact label match)
user_protect_exact = {};
if isfield(cfg, 'protected_terms')
    user_protect_exact = cfg.protected_terms;
end

all_protect_exact = [default_protect_exact, user_protect_exact];

for k = 1:n_terms
    lbl = raw_labels_clean{k};
    for p = 1:length(all_protect_exact)
        % Exact match only — the full cleaned label must equal the pattern
        if strcmp(lbl, all_protect_exact{p})
            protected_idx(k) = true;
            break;
        end
    end
end

% Also document intent in output
if isfield(cfg, 'verbose') && cfg.verbose
    n_prot = sum(protected_idx);
    fprintf('  Protected terms: %d (action terms + clinician-mandated interactions)\n', n_prot);
end

end
