%% Main_Outer_Control_Loop.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

% Based on Li and Todorov (2007): Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system
% Adaptive and dual extensions by Andrew Mathis

% DEPENDENCIES
% iLQG_function.m
% l_cost.m
% Measurement.m
% ContinuousStateDynamics.m
% DiscreteStateDynamics.m
% forward_pass.m
% backward_pass.m
% Todorov_estimator.m
% SPKF_function.m
% simulate_system.m
% finite_difference.m
% boxQP.m
% makePD.m
% pp.m
% sabs.m


%% Clear
clc
clear
close all
rng('default')
tic

%% System inputs
T = 30; % days
dt = 1; % days
N = round(T/dt); % number of time steps

x_hat = [82999999-1000 1000 0 0 0 0 0 0]'; % estimate of initial state
x_true = x_hat; % true initial state
cov_X = 1e-1; % initial state covariance

p_true = [0.0422, 0.3614, 0.2, 0.0018, 0.015, 0.05, 0.25, 0.0422, 0.3614, 0.1, 0.0414, 0.3548, 0.1, 0.0414, 0.3548, 0.3]'; % true parameter values
p_hat = [0.05, 0.5, 0.5, 0.005, 0.05, 0.1, 0.2, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5]'; % estimate of parameter values
cov_P = diag((0.5*p_hat*2/6).^2); % initial parameter covariance

constants = [82999999, 0.0790, 0.0596, 0.0563, 0.0080, 0.0050, 0.173, 0.0370, 0.0552, 0.0159, 0.0242, 15531/82999999]';

u_lims = [0, 1; 0, 1; 0, 1; 0, 1; 0, 1]; % control limits
u_lim_method = 2; % 1: BoxQP, (2): squashing

nu = size(u_lims, 1); % number of controls
nx = size(x_hat, 1); % number of states
np = size(p_hat, 1); % number of parameters
ny = 4; % number of measurements
nv = ny; % measurement noise size, not necessarily same as measurement size (ny)
nw = []; % dynamics noise size, not necessarily same as state size (nx)

u_bar = 0.*rand([nu, N]);  % initial control trajectory if not using seeds [nu N]
Tracking_Trajectory = [];

first_run_seeds = 1; % number of seeds to use for the first iteration, must be >= 1
seed_type = 1; % type of random seeds to generate; (1) random noise, (2) random single value for each control trajectory
u_seed_mean = 0; % mean of random control sequence for seeds
u_seed_cov = 0.25^2; % covariance of random control sequence for seeds
seed_selection_type = 1; % how to select from seeded solutions; (1): best solution, (2): mean of all solutions

%% Algorithm options (see more options at start of iLQG_function, as well as settings in Dynamics and Measurement)
lambda_i = 1;  % initial value for lambda, default = 1
dlambda_i = 1; % initial value for dlambda, default = 1
reset_lambda = 1; % reset lamdba and dlambda to initial values above when initializing each inner loop (0/1)
dyn_noise_reg = 0; % regularize dynamics noise in inner loop? (0/1) - no longer used
regType = 3; % lamdba reg. option to use.
% (1): H_reg = H + lambda*I (Todorov 2005),
% (2): S_reg = S + lambda*I (Todorov 2005),
% (3): H_reg = H + (lambda - min(eig(H)))*I (Todorov 2007),
% (4): H_reg = V*D*V', where [V, D] = eig(H) and D < lambda = lambda (Todorov 2007)

max_du_iterations = 250; % maximum inner loop iLQG iterations to run before terminating
first_run_max_du_iterations = 500; % number of iterations to run on the first inner loop iteration, if only 1 seed
seeds_max_du_iterations = first_run_max_du_iterations; % number of iterations to run for the seeds runs of the first inner loop iteration

augment_states_in_iLQG = 1; % adaptive (0) or dual (1)
augment_states_in_filter = 1; % (1) for adaptive or dual, (0) for non-adaptive
keep_dyn_noise = 0; % keep the state dynamics noise in the true system (0/1)
keep_meas_noise = 0; % keep the measurement noise in the true system (0/1)
keep_dyn_noise_filter = 1; % keep the state dynamics covariance in the filter (0/1)
keep_meas_noise_filter = 1; % keep the measurement covariance in the filter (0/1)
horizon_mode = 2; % (1) shrinking horizon, (2) rolling horizon

verbose = 0; % display status in command window? (0/1)
show_plots_inner_loop = 0; % show plots for each (inner loop) iLQG iteration, inlcuding seed runs? (0/1)
show_plots_outer_loop = 0;
show_seed_plots = 1; % show seed plots after each seed run
show_plots_at_end = 1; % show plots at the end (of the outer loop)? (0/1)
n_sigma_plot = 3; % half range of standard deviations to plot, x_nom +/- n_sigma_plot sigma

% Sigma Point Kalman Filter options
h = sqrt(3); % SPKF h term
AdditiveNoise = 0; % is noise additive Gaussian (0/1)

%% Initialization

if isempty(nw)
    if augment_states_in_iLQG
        nw = nx+np;
    else
        nw = nx;
    end
end

% for if only 1 seed:
l = u_bar; % initial control gain l [nu N]
L = zeros([nu, nx + np*augment_states_in_iLQG, N]); % initial control gain L [nu, nxa, N]

ix = 1:length(x_hat);

if augment_states_in_filter
    ip = (1:length(p_hat)) + ix(end);
    xa_true = [x_true(:, 1); p_true];
    constants_filter = constants;
    cov_xa_hat_filter = blkdiag(cov_X*eye(nx), cov_P*eye(np));
else
    ip = 1;
    xa_true = [x_true(:, 1)];
    constants_filter = [constants; p_hat];
    cov_xa_hat_filter = blkdiag(cov_X*eye(nx));
end

if augment_states_in_iLQG
    cov_xa_hat_iLQG = blkdiag(cov_X*eye(nx), cov_P*eye(np));
else
    cov_xa_hat_iLQG = blkdiag(cov_X*eye(nx));
end

cov_xa_hat_all = cov_xa_hat_filter(:);

R_cov = 1*eye(nv); % measurement noise (co)variance, = 1 as defined from Brownian noise
Q_cov = 1*eye(nw); % dynamics noise (co)variance, = 1 as defined from Brownian noise
sqrtR = chol(R_cov, 'lower');
sqrtQ = chol(Q_cov, 'lower');

if horizon_mode == 1
    T_rem = T:-dt:0;
elseif horizon_mode == 2
    T_rem = T.*ones(1, N+1);
end

% Set lambda and dlambda to initial values
lambda = lambda_i;
dlambda = dlambda_i;

%%
tic
for iter_outer = 1:N-1
    %% Run iLQG

    % display progress
    disp(['Progress: Step #', num2str(iter_outer), ' of ', num2str(N)])

    if iter_outer == 1 && first_run_seeds > 1
        if show_seed_plots
            f3 = figure(3);
            movegui(f3,'south');
            f3.Position = [5 45 1591 345];
        end
        for seed = 1:first_run_seeds
            if seed == 1
                u_bar = zeros(nu, N);
            else
                switch seed_type
                    case 1 % random noise
                        l = u_seed_mean + chol(u_seed_cov)*randn(nu, N); % [nu N]
                        L = zeros([nu, nx + np*augment_states_in_iLQG, N]); % [nu, nxa, N]
                    case 2 % random lines
                        l = u_seed_mean + chol(u_seed_cov)*randn(nu, 1)*ones(1, N); % [nu N]
                        L = zeros([nu, nx + np*augment_states_in_iLQG, N]); % [nu, nxa, N]
                end

                u_bar = l;
            end

            [xnew, unew, l, L, ~, ~, cost, Pw, Pv, converged] = iLQG_function(T_rem(iter_outer), dt, x_hat(:, iter_outer), l, L, u_bar, lambda, dlambda, constants, p_hat(:, iter_outer), cov_xa_hat_iLQG, augment_states_in_iLQG, regType, u_lims, ny, nv, nw, seeds_max_du_iterations, verbose, show_plots_inner_loop, dyn_noise_reg, Tracking_Trajectory, u_lim_method);


            u_seeds(:, :, seed) = unew;
            l_seeds(:, :, seed) = l;
            L_seeds(:, :, :, seed) = L;
            cost_seeds(seed) = sum(cost);
            Pw_seeds(:, :, :, seed) = Pw;
            Pv_seeds(:, :, :, seed) = Pv;

            if show_seed_plots
                figure(3)
                for i = 1:nu
                    subplot(1, nu+1, i)
                    hold on
                    stairs(dt*(1:N), unew(i, :)');
                    hold off
                end
                subplot(1, nu+1, nu+1)
                hold on
                plot(seed, cost_seeds(seed), '*')
                hold off

                drawnow()
            end
        end

        % select final solution from seeds
        switch seed_selection_type
            case 1 % minimum cost
                [lowest_cost, lowest_cost_seed_index] = min(cost_seeds);
                unew = u_seeds(:, :, lowest_cost_seed_index);
                l = l_seeds(:, :, lowest_cost_seed_index);
                L = L_seeds(:, :, :, lowest_cost_seed_index);
                Pw = Pw_seeds(:, :, :, lowest_cost_seed_index);
                Pv = Pv_seeds(:, :, :, lowest_cost_seed_index);
            case 2 % mean of all solutions
                unew = mean(u_seeds, 3);
                l = mean(l_seeds, 3);
                L = mean(L_seeds, 4);
                Pw = mean(Pw_seeds, 4);
                Pv = mean(Pv_seeds, 4);
        end

        u_bar = unew;
        u(:, iter_outer) = unew(:, 1);
    elseif iter_outer == 1
        % use first_run_max_du_iterations
        [xnew, unew, l, L, ~, ~, ~, Pw, Pv, converged] = iLQG_function(T_rem(iter_outer), dt, x_hat(:, iter_outer), l, L, u_bar, lambda, dlambda, constants, p_hat(:, iter_outer), cov_xa_hat_iLQG, augment_states_in_iLQG, regType, u_lims, ny, nv, nw, first_run_max_du_iterations, verbose, show_plots_inner_loop, dyn_noise_reg, Tracking_Trajectory, u_lim_method);
        u(:, iter_outer) = unew(:, 1);
    else
        [xnew, unew, l, L, ~, ~, ~, Pw, Pv, converged] = iLQG_function(T_rem(iter_outer), dt, x_hat(:, iter_outer), l, L, u_bar, lambda, dlambda, constants, p_hat(:, iter_outer), cov_xa_hat_iLQG, augment_states_in_iLQG, regType, u_lims, ny, nv, nw, max_du_iterations, verbose, show_plots_inner_loop, dyn_noise_reg, Tracking_Trajectory, u_lim_method);
        u(:, iter_outer) = unew(:, 1);

    end

    if reset_lambda
        lambda = lambda_i;
        dlambda = dlambda_i;
    end

    %% Simulate true system with noise for one time step

    if augment_states_in_filter
        Q_cov = 1*eye(nx+np); % dynamics noise (co)variance, = 1 as defined from Brownian noise
        sqrtQ = chol(Q_cov, 'lower');
    end
    [y_sim, xa_sim] = simulate_system(dt, xa_true(:, iter_outer), u(:, iter_outer), constants, keep_meas_noise.*sqrtR, keep_dyn_noise.*sqrtQ, augment_states_in_filter, 0, u_lims, u_lim_method);
    xa_true(:, iter_outer + 1) = xa_sim(:, end);
    y_true(iter_outer, :) = y_sim(end, :);

    x_true(:, iter_outer + 1) = xa_true(ix, iter_outer + 1);

    cost_true(iter_outer) = l_cost(xa_true(:, iter_outer+1), u(:, iter_outer), Tracking_Trajectory, u_lims, u_lim_method, constants);


    %% Update state and parameter estimates and covariances

    if augment_states_in_filter
        xa_hat = [x_hat(:, iter_outer); p_hat(:, iter_outer)];
        [~, ~, ~, ~, Fp] = ContinuousStateDynamics(dt, xa_hat, u(:, iter_outer), constants_filter, augment_states_in_filter, u_lims, u_lim_method);
        Pw(nx+1:nx+np, nx+1:nx+np, :) = Fp(1, 1);
    else
        xa_hat = [x_hat(:, iter_outer)];
    end

    Dynamics_F = @(dt, xk, wk)DiscreteStateDynamics(dt, xk, u(:, iter_outer), constants_filter, wk, 0, augment_states_in_filter, 1, 0, u_lims, u_lim_method);
    Measurement_F = @(xk, vk)Measurement(dt, xk, u(:, iter_outer), constants_filter, vk, 0, augment_states_in_filter, 1); % (vk\Gv_sqrtdt)' was previously zero for the PF, as this is what PF sends anyway

    [xa_hat_filtered, P_x_filtered] = SPKF_function(dt, Dynamics_F, Measurement_F, y_true(iter_outer, :), xa_hat, zeros(size(xa_hat)), zeros(size(y_true(iter_outer, :)')), cov_xa_hat_filter, Pw.*keep_dyn_noise_filter, Pv.*keep_meas_noise_filter, h, AdditiveNoise);

    % cost_est(iter_outer) = l_cost(xa_hat, u(:, iter_outer));
    %% Prep for next iLQG iteration


    x_hat(:, iter_outer + 1) = xa_hat_filtered(ix)';
    cost_est(iter_outer) = l_cost(xa_hat_filtered', u(:, iter_outer), Tracking_Trajectory, u_lims, u_lim_method, constants);

    % Set warm start control policy variables
    if horizon_mode == 1
        l = l(:, 2:end);
        L = L(:, :, 2:end);
        u_bar = unew(:, 2:end);
        Tracking_Trajectory = Tracking_Trajectory(:, 2:end);
    elseif horizon_mode == 2
        u_bar = unew(:, 2:end);
        u_bar(:, N) = unew(:, end);
    end

    if augment_states_in_filter
        p_hat(:, iter_outer + 1) = xa_hat_filtered(ip)';
    else
        p_hat(:, iter_outer + 1) = p_hat(:, iter_outer);
    end

    cov_xa_hat_filter = P_x_filtered;
    if augment_states_in_iLQG == 0
        cov_xa_hat_iLQG = cov_xa_hat_filter(1:nx, 1:nx);
    else
        cov_xa_hat_iLQG = cov_xa_hat_filter;
    end
    cov_xa_hat_all(:, iter_outer + 1) = cov_xa_hat_filter(:); % for plotting

    xa_hat = [x_hat; p_hat]; % for plotting

    %% Plots

    if show_plots_outer_loop

        f2 = figure(2);
        if iter_outer == 1
            movegui(f2,'south');
            f2.Position = [5 45 1591 345];
        end
        % plot true and estimated cost
        subplot(2, round((1 + nx + np*augment_states_in_filter + nu)/2, 0) + 1, 1)
        stairs(1: iter_outer+1, [cost_true, cost_true(end)]', 'g');
        hold on
        stairs(1: iter_outer+1, [cost_est, cost_est(end)]', 'r');
        hold off

        for i_plot = 1:nx + np*augment_states_in_filter

            subplot(2, round((1 + nx + np*augment_states_in_filter + nu)/2, 0) + 1,  i_plot + 1)

            % plot true and estimated augments state vectors
            plot(1: iter_outer + 1, xa_hat(i_plot, :), 'r-', 1: iter_outer + 1, xa_true(i_plot, :), 'g-')
            if augment_states_in_filter
                if i_plot < ip(1)
                    axis_label = ['State ', num2str(i_plot)];
                else
                    axis_label = ['Parameter ', num2str(i_plot-ix(end))];
                end
            else
                axis_label = ['State ', num2str(i_plot)];
            end
            ylabel(axis_label)

            % plot augmented covariances
            hold on
            Sigmas_plot = n_sigma_plot*sqrt(cov_xa_hat_all(ip(end)*(i_plot-1)+i_plot, :));
            plot(1: iter_outer + 1, xa_hat(i_plot, :) + Sigmas_plot, 'r--', 1: iter_outer + 1, xa_hat(i_plot, :) - Sigmas_plot, 'r--')
        end
        % plot actual control actions
        for i_plot = 1:nu
            subplot(2, round((1 + nx + np*augment_states_in_filter + nu)/2, 0) + 1,  nx + np*augment_states_in_filter + i_plot  + 1)
            u_squashed = (u_lims(:, 2) - u_lims(:, 1))/2.*tanh(u) + (u_lims(:, 2) + u_lims(:, 1))/2;
            stairs(1: iter_outer+1, [u_squashed(i_plot, :), u_squashed(i_plot, end)]')
            ylim([0 1])
            axis_label = ['Control ', num2str(i_plot)];
            ylabel(axis_label)
        end
        drawnow
    end

end

%% Last time step costs
cost_est(iter_outer+1) = l_cost(xa_hat(:, iter_outer+1), NaN(nu, 1), Tracking_Trajectory, u_lims, u_lim_method, constants);
cost_true(iter_outer+1) = l_cost(xa_true(:, iter_outer+1), NaN(nu, 1), Tracking_Trajectory, u_lims, u_lim_method, constants);

%%
if show_plots_outer_loop
    subplot(1, 1 + nx + np*augment_states_in_filter + nu, 1)
    hold on
    plot(1:iter_outer+1, sum(cost_true).*ones(size(cost_true)), '--g', 1:iter_outer+1, sum(cost_est).*ones(size(cost_est)), '--r')
end

cost = sum(cost_true)
time = toc

%% Plot at end

%% Plots

if show_plots_at_end

    % plot true and estimated cost
    figure
    plot((0:iter_outer+1)*dt, [0, cost_true], 'go')
    ylabel('Cost')

    % cost total lines
    hold on
    plot((1:iter_outer+1)*dt, sum(cost_true).*ones(size(cost_true)), '--g')

    % plot true and estimated augments state vectors
    figure
    for i_plot = 1:nx + np*augment_states_in_filter
        subplot(1, nx+np, i_plot)
        plot((0: iter_outer)*dt, xa_hat(i_plot, :), 'g-')
        hold on
        plot((0: iter_outer)*dt, xa_true(i_plot, :), 'k-')
        hold off
        if augment_states_in_filter
            if i_plot < ip(1)
                axis_label = ['State ', num2str(i_plot)];
            else
                axis_label = ['Parameter ', num2str(i_plot-ix(end))];
            end
        else
            axis_label = ['State ', num2str(i_plot)];
        end
        ylabel(axis_label)

        % plot augmented covariances
        hold on
        Sigmas_plot = n_sigma_plot*sqrt(cov_xa_hat_all(ip(end)*(i_plot-1)+i_plot, :));
        plot((0: iter_outer)*dt, xa_hat(i_plot, :) + Sigmas_plot, 'g--', (0: iter_outer)*dt, xa_hat(i_plot, :) - Sigmas_plot, 'g--')
    end

    % plot actual control actions
    if isempty(u_lims)
        figure
        for i_plot = 1:nu
            subplot(1, nu, i_plot)
            stairs((0: iter_outer)*dt, [u(i_plot, :), u(i_plot, end)]', 'g')
            axis_label = ['Control ', num2str(i_plot)];
            ylabel(axis_label)
        end

    elseif u_lim_method == 2 && ~isempty(u_lims)
        figure
        for i_plot = 1:nu
            subplot(1, nu, i_plot)
            u_squashed = (u_lims(:, 2) - u_lims(:, 1))/2.*tanh(u) + (u_lims(:, 2) + u_lims(:, 1))/2;
            stairs((0: iter_outer)*dt, [u(i_plot, :), u(i_plot, end)]', '--g')
            hold on
            stairs((0: iter_outer)*dt, [u_squashed(i_plot, :), u_squashed(i_plot, end)]', 'k')
            hold off
            axis_label = ['"Pre-control" and control ', num2str(i_plot)];
            ylabel(axis_label)
        end

    end



end
