function [y, xa] = simulate_system(dt, xa0, U, c, sqrtR, sqrtQ, augment_states, lambda, u_lims, u_lim_method)
%SIMULATE_SYSTEM  Simulate one SCI rehabilitation trajectory.
%
%  Propagates the true patient state forward using the ground-truth
%  synthetic dynamics (not the SINDy model), adds measurement noise,
%  and returns the observed trajectory.
%
%  This is called by Outer_Control_Loop to generate the "true" observations
%  that the SPKF filter uses to update the state/parameter estimates.

v = sqrtR * randn([size(sqrtR, 1), 1]);
y(:, 1) = Measurement(dt, xa0, U(:, 1), c, v, false, augment_states, 0);

U(:, end+1) = U(:, end);  % extend final control

xa(:, 1) = xa0;
for i = 1:size(U, 2) - 1
    if i > 1
        xa0 = xa(:, i);
    end

    w       = sqrtQ * randn([size(sqrtQ, 1), 1]);
    x_next  = DiscreteStateDynamics(dt, xa0, U(:,i), c, w, false, ...
                                     augment_states, 0, lambda, u_lims, u_lim_method);
    xa(:, i+1) = x_next;

    v = sqrtR * randn([size(sqrtR, 1), 1]);
    y(:, i+1) = Measurement(dt, xa(:,i+1), U(:,i+1), c, v, false, augment_states, 0);
end

y = y';

end
