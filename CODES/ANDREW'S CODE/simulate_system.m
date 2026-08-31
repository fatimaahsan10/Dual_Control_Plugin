%% forward_pass.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

function [y, xa] = simulate_system(dt, xa0, U, c, sqrtR, sqrtQ, augment_states, lambda, u_lims, u_lim_method)

v = sqrtR*randn([length(sqrtR), 1]);

y(:, 1) = Measurement(dt, xa0, U(:, 1), c, v, false, augment_states, 0);

U(:, end+1) = U(:, end);

xa(:, 1) = xa0;
for i = 1:size(U, 2)-1
    if i > 1
        xa0 = xa(:, i);
    end

    w = sqrtQ*randn([length(sqrtQ), 1]);
    x_ode = DiscreteStateDynamics(dt, xa0, U(:, i), c, w, false, augment_states, 0, lambda, u_lims, u_lim_method);
    xa(:, i+1) = x_ode;

    if ~isreal(x_ode)
        warning('Imaginary system dynamics!')
    end

    v = sqrtR*randn([length(sqrtR), 1]);
    y(:, i+1) = Measurement(dt, xa(:, i+1), U(:, i+1), c, v, false, augment_states, 0);
end

y = y';

end