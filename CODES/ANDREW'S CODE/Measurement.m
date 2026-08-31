%% Measurement.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

% Based on Li and Todorov (2007): Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system


function yk = Measurement(dt, xk, u, c, v, G_only_col, augment_states, v_from_filter) % also function of uk, theta
% SIDARTHE
% 12345678
%g = [xk(1, :); xk(2, :); xk(3, :); xk(4, :); xk(5, :); xk(6, :); xk(7, :); xk(8, :)];
g = [xk(3, :); xk(5, :); xk(6, :); xk(8, :)];

Np1 = size(xk, 2);
if size(u, 2) == 1 && Np1 > 1
    u = u*ones(1, Np1);
end

ny = size(g, 1);
nv = size(v, 1);

if v_from_filter == 0
    for j = 1:size(xk, 2)
        G(:, :, j) = 1e1.*diag([1 0.75, 0.5, 0.25]); % [ny by nv]
        Gv(:, j) = G(:, :, j)*v(:, j);
    end
    noise = Gv*sqrt(dt);
else
    noise = v;
end

yk =  g + noise; 

if G_only_col > 0
    yk = G(:, G_only_col, :);
end

end