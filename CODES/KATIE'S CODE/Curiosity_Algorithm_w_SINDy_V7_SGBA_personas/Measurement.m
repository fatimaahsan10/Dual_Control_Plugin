function yk = Measurement(dt, xk, u, c, v, G_only_col, augment_states, v_from_filter)
%MEASUREMENT  Observation function for SCI rehabilitation state estimation.
%
%  In rehabilitation, we observe functional outcome scores at each session.
%  Observations: [SCIM_norm; BBS_norm] — the two dynamic states.
%  (AIS and other static attributes are known at admission and fixed.)
%
%  Measurement model: y = g(x) + G*v*sqrt(dt)
%  where G encodes measurement noise scaling per instrument.

%% ── Observation function g(x) ───────────────────────────────────────────

% Observe SCIM (x1) and BBS (x2) directly
g = [xk(1, :);   % SCIM_norm
     xk(2, :)];  % BBS_norm

Np1 = size(xk, 2);
if size(u, 2) == 1 && Np1 > 1
    u = u * ones(1, Np1);
end

ny = size(g, 1);   % 2 observations
nv = size(v, 1);

%% ── Noise ────────────────────────────────────────────────────────────────

if v_from_filter == 0
    for j = 1:Np1
        % Measurement noise scaling:
        %   SCIM has higher inter-rater reliability (lower noise) than BBS
        G(:, :, j) = diag([0.03, 0.05]);   % [ny x nv]
        Gv(:, j)   = G(:, :, j) * v(:, j);
    end
    noise = Gv * sqrt(dt);
else
    noise = v;
end

yk = g + noise;

if G_only_col > 0
    yk = G(:, G_only_col, :);
end

end
