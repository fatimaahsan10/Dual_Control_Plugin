function [nmae, rmse, sim_trajectories] = validate_sindy_model( ...
    Xi, X_val, A_val, Y_val, lib_labels, cfg)
%VALIDATE_SINDY_MODEL  Validate the identified SINDy model by forward
%  simulation on held-out patient trajectories and computing NMAE/RMSE.
%
%  The validation criterion from the manuscript (Section IV-B):
%    Accept model if NMAE < 1.0 (average deviation < full dynamic range)
%
%  INPUTS
%    Xi              [n_terms x n_states]  sparse coefficient matrix (physical units)
%    X_val           [n_states x n_sessions x n_val]   validation states
%    A_val           [n_actions x n_sessions x n_val]  validation actions
%    Y_val           [n_outcomes x n_sessions x n_val] validation outcomes
%    lib_labels      {1 x n_terms}  library term labels
%    cfg             configuration struct
%
%  OUTPUTS
%    nmae            scalar  normalised mean absolute error
%    rmse            scalar  root mean squared error
%    sim_trajectories struct  simulated vs actual trajectories for plotting

[n_states, n_sessions, n_val] = size(X_val);

X_sim = zeros(n_states, n_sessions, n_val);

for p = 1:n_val
    % Initialise at true first observation
    x_sim = X_val(:, 1, p);
    X_sim(:, 1, p) = x_sim;

    for s = 1:n_sessions - 1
        a_s   = A_val(:, s, p);

        % Build one-row library vector for current (x_sim, a_s)
        % NOTE: Theta is in physical units (no external normalisation);
        %       build_library_row must match build_sindy_library column order.
        z         = [x_sim; a_s]';          % 1 x (n_states + n_actions)
        theta_row = build_library_row(z, cfg);  % 1 x n_terms, physical units

        % Predicted derivative: dX = theta_row * Xi  (physical units)
        dx_pred = (theta_row * Xi)';        % n_states x 1

        % Euler integration (dt = 1 week)
        x_sim = x_sim + dx_pred;
        x_sim = max(0, min(1, x_sim));      % enforce [0,1] bounds

        X_sim(:, s+1, p) = x_sim;
    end
end

%% ── Compute error metrics ────────────────────────────────────────────────

% Focus on the two dynamic states: SCIM (1) and BBS (2)
dynamic_idx = [1, 2];

X_true_dyn = X_val(dynamic_idx, :, :);
X_sim_dyn  = X_sim(dynamic_idx, :, :);
err        = X_true_dyn - X_sim_dyn;

% RMSE
rmse = sqrt(mean(err(:).^2));

% NMAE: normalised by the dynamic range of true observations
dyn_range = max(X_true_dyn(:)) - min(X_true_dyn(:));
if dyn_range < eps
    dyn_range = 1;
end
nmae = mean(abs(err(:))) / dyn_range;

%% ── Package for plotting ─────────────────────────────────────────────────

sim_trajectories.X_true     = X_val;
sim_trajectories.X_sim      = X_sim;
sim_trajectories.n_val      = n_val;
sim_trajectories.n_sessions = n_sessions;
sim_trajectories.state_idx  = dynamic_idx;  % states plotted

end

%% ── Local helpers ────────────────────────────────────────────────────────

function theta_row = build_library_row(z, cfg)
%BUILD_LIBRARY_ROW  Reconstruct one library row for a given observation z.
%  Matches the column ordering in build_sindy_library.m exactly.
%  Returns physical-unit values (no normalisation applied).

n_vars = length(z);
cols   = {1};                       % constant term

% Linear
for i = 1:n_vars
    cols{end+1} = z(i);             %#ok<AGROW>
end

% Polynomial degree 2
if cfg.poly_order >= 2
    for i = 1:n_vars
        cols{end+1} = z(i)^2;       % squared
    end
    if cfg.include_cross
        for i = 1:n_vars
            for j = i+1:n_vars
                cols{end+1} = z(i)*z(j);   % cross-product
            end
        end
    end
end

if cfg.poly_order >= 3
    n_states_approx = length(z) - 3;   % rough: total vars - n_actions
    for i = 1:min(n_states_approx, 2)
        cols{end+1} = z(i)^3;
    end
end

if cfg.include_trig
    for i = 1:n_vars
        cols{end+1} = sin(z(i));
        cols{end+1} = cos(z(i));    %#ok<AGROW>
    end
end

theta_row = [cols{:}];
end
