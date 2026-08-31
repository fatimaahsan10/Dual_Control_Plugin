%% DiscreteStateDynamics.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

function xkplus1 = DiscreteStateDynamics(dt, xa, u, c, w, F_only_col, augment_states, w_from_filter, lambda, u_lims, u_lim_method)
% Options
time_step_method = 1; % Method for performing dynamics steps; (1): Euler, (2): Runge-Kutta 4

%% Adjust length of vectors (do not modify)

Nplus1 = size(xa, 2);

if size(w, 2) == 1 && Nplus1 > 1
    w = w*ones(1, Nplus1);
end

if size(u, 2) == 1 && Nplus1 > 1
    u = u*ones(1, Nplus1);
end

%% Initialize Runge-Kutta 4
xa_original = xa;
added_rk4 = 0;
for i = 1:1+3*(time_step_method == 2)

    xa = xa_original + added_rk4;
    %%

    [fa, nx, np, F, Fp] = ContinuousStateDynamics(dt, xa, u, c, augment_states, u_lims, u_lim_method);

    %% Runge-Kutta 4

    if time_step_method == 2
    switch i
        case 1
            k1 = fa;
            added_rk4 = dt*k1/2; % for next iteration:
            if time_step_method == 1
                break
            end
        case 2
            k2 = fa;
            added_rk4 = dt*k2/2; % for next iteration:
        case 3
            k3 = fa;
            added_rk4 = dt*k3; % for next iteration:
        case 4
            k4 = fa;
    end
    end
end

if time_step_method == 2
        fa = 1/6*(k1 + 2*k2 + 2*k3 + k4); % no dt as h as multiplying later
end

%% Noise
nw = size(w, 1);

if w_from_filter == 0
    for i = 1:Nplus1
        Fa(:, :, i) = blkdiag(F, Fp) ; % [nxa by nw]
        Fw(:, i) = Fa(:, :, i)*w(:, i);
    end
    noise = Fw*sqrt(dt);
else
    noise = w;
end

%% Set output

xkplus1 = xa_original + fa*dt + noise;

if any(xkplus1<0)
    %warning('Negative state(s)')
end

if ~isreal(xkplus1)
    warning('Imaginary system dynamics!')
end

if F_only_col > 0
    xkplus1 = Fa(:, F_only_col, :);
end



end