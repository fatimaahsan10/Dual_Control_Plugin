function [xdot, nx, np, F, Fp] = ContinuousStateDynamics(dt, xa, u, c, augment_states, u_lims, u_lim_method)
%CONTINUOUSSTATEDYNAMICS  SCI rehabilitation dynamics using SINDy model.
%
%  Replaces the SIDARTHE epidemic model with SINDy-identified rehabilitation
%  dynamics. Signature kept identical to the original for iLQG compatibility.
%
%  STATE VECTOR (nx = 6, all normalised to [0,1]):
%    x1: SCIM_norm   x2: BBS_norm    x3: AIS (fixed)
%    x4: Age_norm    x5: DPI_norm    x6: Caregiver (fixed)
%
%  ACTION VECTOR (nu = 3):
%    u1: Intensity   u2: Modality    u3: Frequency
%
%  AUGMENTED PARAMETERS (when augment_states = 1):
%    xa(nx+1 : nx+np) = uncertain SINDy coefficients being estimated online.
%    The dual iLQG drives curiosity toward actions that maximise information
%    gain about these parameters (reduces Fp uncertainty).
%
%  CONSTANTS STRUCT c (fields):
%    .Xi           [n_terms x nx]  full SINDy coefficient matrix (physical)
%    .lib_cfg      struct          library config (poly_order, include_cross, ...)
%    .swat_ceiling scalar          patient-specific recovery ceiling [0,1]
%    .xi_idx       [np x 2]        [term_idx, state_idx] for uncertain params
%    .F_dyn        scalar          dynamics noise scale (default 1e-3)
%    .F_param      scalar          parameter noise scale (default 1e-4)

%% ── Setup ────────────────────────────────────────────────────────────────

global SINDY_MODEL;

Nplus1 = size(xa, 2);
nx     = 6;

% Control squashing
if u_lim_method == 2 && ~isempty(u_lims)
    u = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u) + ...
        (u_lims(:,2) + u_lims(:,1))/2;
end

% Read SINDy model from global (avoids struct/vector concat issue in iLQG)
Xi_full = SINDY_MODEL.Xi;
lib_cfg = SINDY_MODEL.lib_cfg;
swat    = SINDY_MODEL.swat_ceiling;
xi_idx  = SINDY_MODEL.xi_idx;
F_dyn   = SINDY_MODEL.F_dyn;
F_param = SINDY_MODEL.F_param;

%% ── Unpack augmented state ───────────────────────────────────────────────

if augment_states
    x_state = xa(1:nx,    :);
    xi_vec  = xa(nx+1:end, :);
    np      = size(xi_vec, 1);
else
    x_state = xa(1:nx, :);
    xi_vec  = [];
    np      = 0;
end

%% ── SINDy dynamics per column ────────────────────────────────────────────

% Defensive check: np from xa must match xi_idx rows
if augment_states && np ~= size(xi_idx, 1)
    error(['ContinuousStateDynamics: np from xa (%d) does not match ' ...
           'xi_idx rows (%d).\nThis usually means xa was double-augmented ' ...
           '(iLQG appended p_hat to an already-augmented x0).\n' ...
           'Pass only x_hat (nx rows) as x0 to iLQG_function, not [x_hat; p_hat].'], ...
           np, size(xi_idx,1));
end

f = zeros(nx, Nplus1);

for col = 1:Nplus1
    x_col = x_state(:, col);
    u_col = u(:, min(col, size(u,2)));

    % Rebuild Xi: start from identified model, overwrite uncertain entries
    Xi_col = Xi_full;
    if augment_states
        for k = 1:np
            t_idx = xi_idx(k, 1);
            s_idx = xi_idx(k, 2);
            Xi_col(t_idx, s_idx) = xi_vec(k, col);
        end
    end

    % Evaluate library and compute derivatives
    theta = build_sindy_row(x_col, u_col, lib_cfg);   % [1 x n_terms]
    f_col = (theta * Xi_col)';                          % [nx x 1]

    % Static states have no dynamics
    f_col(3:6) = 0;

    % Enforce recovery ceiling and floor
    for s = 1:2
        if x_col(s) <= 0 && f_col(s) < 0
            f_col(s) = 0;
        elseif x_col(s) >= swat && f_col(s) > 0
            f_col(s) = 0;
        end
    end

    f(:, col) = f_col;
end

%% ── Parameter dynamics (zero-mean random walk) ───────────────────────────

if augment_states
    fp = zeros(np, Nplus1);
else
    fp = [];
end

xdot = [f; fp];

%% ── Noise matrices ───────────────────────────────────────────────────────

F  = F_dyn   .* eye(nx);
if augment_states
    Fp = F_param .* eye(np);
else
    Fp = [];
end

end

%% ── Local: one-row SINDy library ─────────────────────────────────────────

function theta_row = build_sindy_row(x, u_vec, cfg)
z      = [x(:); u_vec(:)]';
n_vars = length(z);
cols   = {1};

for i = 1:n_vars
    cols{end+1} = z(i); %#ok<AGROW>
end

if cfg.poly_order >= 2
    for i = 1:n_vars
        cols{end+1} = z(i)^2;
    end
    if cfg.include_cross
        for i = 1:n_vars
            for j = i+1:n_vars
                cols{end+1} = z(i)*z(j);
            end
        end
    end
end

if cfg.poly_order >= 3
    for i = 1:min(length(x), 2)
        cols{end+1} = z(i)^3;
    end
end

if isfield(cfg,'include_trig') && cfg.include_trig
    for i = 1:n_vars
        cols{end+1} = sin(z(i));
        cols{end+1} = cos(z(i)); %#ok<AGROW>
    end
end

theta_row = [cols{:}];
end
