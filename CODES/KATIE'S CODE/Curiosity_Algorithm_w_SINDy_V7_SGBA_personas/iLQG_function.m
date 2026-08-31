function [xa0, u_bar, l, L, lambda, dlambda, costnew, Pw, Pv, converged] = iLQG_function(T, dt, x0, l, L, u_bar, lambda, dlambda, constants, p_hat, cov_xa_hat_0, augment_states, regType, u_lims, ny, nv, nw, max_du_iterations, verbose, show_plots, dyn_noise_reg, Tracking_Trajectory, u_lim_method)

% iLQG_function.m
% Author: Andrew Mathis

% Based on:
% - Li, W., & Todorov, E. (2007). Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system. International Journal of Control, 80(9), 1439–1453. https://doi.org/10.1080/00207170701364913
% - the code associated with Tassa, Y., Mansard, N., & Todorov, E. (2014). Control-limited differential dynamic programming. In Proceedings - IEEE International Conference on Robotics and Automation (pp. 1168–1175). Institute of Electrical and Electronics Engineers Inc. https://doi.org/10.1109/ICRA.2014.6907001

% Inputs:
% T - finial time
% dt - time step
% x0 - initial state estimate
% l & L - initial gain terms
% u_bar - initial control trajectory
% lambda - initial value of the regularization parameter
% dlambda - initial step size for the regularization parameter
% constants - vector of constants for dynamics
% p_hat - vector of estimated parameters for dynamics
% cov_xa_hat_0 - initial state covariance
% augment_states - (0) for normal iLQG, (1) for dual iLQG
% regType - regularization option to use. (1) H_reg = H + lambda*I (Todorov 2005), (2) S_reg = S + lambda*I (Todorov 2005), (3) H_reg = H + (lambda - min(eig(H)))*I
% ulims - control limits, form: [u1_min, u1_max; u2_min, u2_max; ...]
% ny - length of measurement vector
% nv - length of measurement Brownian noise vector
% nw - length of dynamics Brownian noise vector
% max_du_iterations - max. iLQG iterations to run
% verbose - option to print messages to command window, (true) or (false)
% show_plots - option to plot states, parameters, cost, and lambda; (true) or (false)

% Outputs:
% xa0 - final state estimate trajectory
% u_bar - final locally optimal control trajectory
% l, L - control policy gains
% lambda, dlambda - regularization parameter final values
% converged - convergance flag, (0): did not converge, (1): min. gradient tolerence, (2): min. function tolerence, (3): max. iterations
% gradient, (2): cost gradient, (3): max iterations reached, (-1): failed as lambda > lambda_max, (-2):
% failed
% in line search forward pass

%% Options
full_DDP = 0;                   % (1) option not incorporated into iLQG changes from iLQR

dlambda_0 = 1.6;             % lambda scaling factor, default 2
lambdaMax = 1e12;               % lambda maximum value, default 1e12
lambdaMin = 1e-6;               % below this value lambda = 0

serial = true;                  % use serial line-search (true) or parallel (false) - parallel not incorporated
Alpha = 10.^linspace(0,-3,11);  % backtracking coefficients
tolFun = 1e-4;                  % cost reduction exit criterion
tolGrad = 1e-3;                 % gradient exit criterion
zMin = 0;                       % minimal accepted reduction ratio

%% initialize

[nu, N] = size(u_bar);


t = 0:dt:T;                     % length should be N+1

nx = size(x0, 1);
np = size(p_hat, 1);
if augment_states
    nxa = nx + np;
else
    nxa = nx;
end

if augment_states
    xa0 = [x0; p_hat];
else
    xa0 = [x0];
    constants = [constants; p_hat];
end

R_cov = 1*eye(nv); % measurement noise (co)variance, = 1 as defined from Brownian noise
Q_cov = 1*eye(nw); % dynamics noise (co)variance, = 1 as defined from Brownian noise
sqrtR = chol(R_cov, 'lower');
sqrtQ = chol(Q_cov, 'lower');

np = size(p_hat, 1);

ix = 1:nx;
ip = (1:np) + ix(end);

j = 1;
flgChange = 1;
cost = [];
costs_saved = [];
lambdas_saved = lambda;

if u_lim_method == 1
    u_lims_boxQP = u_lims;
elseif u_lim_method == 2
    u_lims_boxQP = [];
end

converged = 0;

if show_plots
    figure(1) % to plot on correct outer loop figure
end

while j < max_du_iterations
    %% Roll out initial trajectory and get future measurements based on p_hat

    %u_bar = (u_lims_squash(:, 2) - u_lims_squash(:, 1))/2.*tanh(u_bar) + (u_lims_squash(:, 2) + u_lims_squash(:, 1))/2;

    if size(xa0, 2) == 1  % x0 is a vector, as it is for the first iteration

        [y_bar, xa_bar] = simulate_system(dt, xa0, u_bar, constants, 0.*sqrtR, 0.*sqrtQ, augment_states, dyn_noise_reg*lambda, u_lims, u_lim_method); % estimated no-noise measurements
        y_bar = y_bar';

    elseif size(xa0, 2) == N+1
        xa_bar = xa0;

        y_bar = Measurement(dt, xa0, [u_bar NaN(nu, 1)], constants, zeros(nv, N+1), false, augment_states, 0);

    else
        error("Error: xa0 incorrect size.")
    end


    %% FORWARD PASS
    if flgChange

        [A, B, c, Cx, Cu, d, Dx, Du, E, F, fxx, fxu, fuu, q0, q, Q, r, R, P]   = forward_pass(xa_bar, [u_bar NaN(nu,1)], constants, @DiscreteStateDynamics, @Measurement, @l_cost, dt, full_DDP, nw, nv, ny, augment_states, dyn_noise_reg, lambda, Tracking_Trajectory, u_lims, u_lim_method);
        cost = l_cost(xa_bar, [u_bar NaN(nu,1)], Tracking_Trajectory, u_lims, u_lim_method, constants); % cost of initial trajectory
        flgChange   = 0;

        if j == 1
            costs_saved = sum(cost);
        end

        if verbose
            disp('Forward pass complete')
        end
    end

    %% ESTIMATOR

    [~, ~, K, m_x, m_e, Sigma_x, Sigma_e, Sigma_xe, Sigma_ex]  = Todorov_estimator(A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, xa_bar, y_bar, u_bar, 0.*xa0(:, 1), cov_xa_hat_0, dt, constants, sqrtR, augment_states);

    if verbose
        disp('Estimator complete')
    end

    %% BACKWARD PASS
    backPassDone   = 0;
    while ~backPassDone

        if verbose
            disp('Running backward pass...')
        end

        [diverge, l, L, s0_a] = backward_pass(A, B, c, Cx, Cu, d, Dx, Du, E, F, K, fxx, fxu, fuu, q0, q, Q, r, R, P,lambda,regType,u_lims_boxQP,u_bar, nv);

        if diverge ~= 0
            lambda_old = lambda;

            dlambda   = max(dlambda * dlambda_0, dlambda_0);
            lambda    = max(lambda * dlambda, lambdaMin);

            if verbose

                if diverge > 0
                    fprintf('Cholesky failed at timestep %d.\n',diverge);
                    disp(['Increasing lambda from ', num2str(lambda_old), ' to ', num2str(lambda)])
                else
                    fprintf('BoxQP failed at timestep %d.\n',-1*diverge);
                    disp(['Increasing lambda from ', num2str(lambda_old), ' to ', num2str(lambda)])
                end
            end

            if lambda > lambdaMax

                if verbose
                    disp('lambda > lambdaMax')
                end
                converged = -1;
                warning('Warning: lambda > lambdaMax')
                break;
            end

            continue
        end
        backPassDone      = 1;

        if verbose
            disp('Backward pass complete')
        end
    end

    % check for termination due to small gradient
    g_norm         = mean(max(abs(l) ./ (abs(u_bar)+1),[],1));

    if g_norm < tolGrad && lambda < 1e-5
        dlambda   = min(dlambda / dlambda_0, 1/dlambda_0);
        lambda    = lambda * dlambda * (lambda > lambdaMin);
        converged = 1;
        if verbose
            fprintf('\nSUCCESS: gradient norm < tolGrad\n');
        end
        break;
    end

    %% LINE SEARCH
    fwdPassDone = 0;

    if verbose
        disp('Starting line search')
    end

    %             du = zeros(nu, N);
    %             xnew = zeros(nxa, N+1);

    if serial  % line-search
        for alpha = Alpha
            if verbose
                disp(['alpha = ', num2str(alpha)])
            end

            %             for k = 1:N
            %                 du(:, k) = alpha*l(:, k) + L(:, :, k)*dx_hat_est(:, k);
            %             end
            %
            %             unew = u_bar + du;
            %
            %                 [y_bar_sim, xa_bar_sim] = simulate_system(dt, xa_bar(:, k) + dx_hat_est(:, k), unew(:, k), constants, 0.*sqrtR, 0.*sqrtQ, augment_states, dyn_noise_reg*lambda);
            %                 xnew(:, k+1) = xa_bar_sim(:, end);

            du = zeros(nu, N);
            unew = zeros(nu, N);
            x_hat = zeros(nxa, N+1);
            x = zeros(nxa, N+1);
            xnew = zeros(nxa, N+1);
            y = zeros(ny, N+1); % correct length?
            Pw = zeros(nxa, nxa, N);
            Pv = zeros(ny, ny, N);

            xnew(:, 1) = xa0(:, 1);
            for k = 1:N
                if k == 1
                    x_hat(:, k) = zeros(nxa, 1);
                    x(:, k) = zeros(nxa, 1);
                end

                % calculate locally optimal control action
                %du(:, k) = alpha*l(:, k) + L(:, :, k)*x_hat(:, k);
                du(:, k) = alpha*l(:, k) + L(:, :, k)*(xnew(:, k) - xa_bar(:, k)); % from 2014 (7b)
                unew(:, k) = u_bar(:, k) + du(:, k);

                % new May 27, 2022:
                %                 if ~isempty(u_lims_squash)
                %                     unew(:, k) = min(unew(:, k), u_lims_squash(:, 2)); % impose max limit
                %                     unew(:, k) = max(unew(:, k), u_lims_squash(:, 1)); % impose min limit
                %                 end

                %unew = (u_lims_squash(:, 2) - u_lims_squash(:, 1))/2.*tanh(unew) + (u_lims_squash(:, 2) + u_lims_squash(:, 1))/2;

                %                 xnew(:, k) = xa_bar(:, k) + x_hat(:, k);
                %
                [y_bar_sim, xa_bar_sim] = simulate_system(dt, xnew(:, k), unew(:, k), constants, 0.*sqrtR, 0.*sqrtQ, augment_states, dyn_noise_reg*lambda, u_lims, u_lim_method);
                xnew(:, k+1) = xa_bar_sim(:, end);

                %y_data  = Measurement(dt, xnew(:, k), unew(:, k), constants, zeros(nv, N+1), false, augment_states, 0);
                %                 y = y_bar_sim(end, :)' - y_bar(:, k);

                for i = 1:nw
                    C(:, i) = c(:, i, k) + Cx(:, :, k, i)*x(:, k) + Cu(:, :, k, i)*du(:, k);
                    Pw(:, :, k) = Pw(:, :, k) + C(:, i)*C(:, i)';
                end

                % Calculate measurement noise covariance
                for i = 1:nv
                    D(:, i) = d(:, i, k) + Dx(:, :, k, i)*x(:, k) + Du(:, :, k, i)*du(:, k);
                    Pv(:, :, k) = Pv(:, :, k) + D(:, i)*D(:, i)';
                end

                zeta = sqrtQ*randn([length(sqrtQ), 1]);
                eta = sqrtR*randn([length(sqrtR), 1]);

                x(:, k+1) = A(:, :, k) * x(:, k) + B(:, :, k) * du(:, k) + C * zeta;
                y(:, k) = F(:, :, k) * x(:, k) + E(:, :, k) * du(:, k) + D * eta;

                % Use estimator to get next x_hat
                x_hat(:, k+1) = A(:, :, k)*x_hat(:, k) + B(:, :, k)*du(:, k) + K(:, :, k)*(y(:, k) - F(:, :, k)*x_hat(:, k) - E(:, :, k)*du(:, k));
                %xnew(:, k) = xa_bar(:, k) + x_hat(:, k);
            end

            [y_bar_sim, xa_bar_sim] = simulate_system(dt, xnew(:, k), unew(:, k), constants, 0.*sqrtR, 0.*sqrtQ, augment_states, dyn_noise_reg*lambda, u_lims, u_lim_method);
            xnew(:, k+1) = xa_bar_sim(:, end);

            %xnew(:, k+1) = xa_bar(:, k+1) + x_hat(:, k+1);

            unew(isnan(unew)) = 0;

            unew = [unew, NaN(nu, 1)];
            costnew = l_cost(xnew, unew, Tracking_Trajectory, u_lims, u_lim_method, constants);

            dcost    = sum(cost(:)) - sum(costnew(:));
            expected = -alpha*(s0_a(1) + alpha*s0_a(2));
            if verbose
                disp(['Cost change: ', num2str(dcost), ', expected: ', num2str(expected)])
            end

            if expected > 0
                z = dcost/expected;
            else
                z = sign(dcost);
                warning('non-positive expected reduction: should not occur');
            end
            if (z > zMin)
                if verbose
                    disp(['z = ', num2str(z), ' therefore z > z_min'])
                end

                fwdPassDone = 1;
                break;
            end

            %             if dcost > 0 % AM added to test removal of above section
            %                 fwdPassDone = 1;
            %                 break;
            %             end

        end

    end

    if ~fwdPassDone
        alpha = NaN; % signals failure of forward pass
    end

    %% ACCEPT x_new, u_new or not

    if fwdPassDone

        % decrease lambda
        lambda_old = lambda;

        dlambda   = min(dlambda / dlambda_0, 1/dlambda_0);
        lambda    = lambda * dlambda * (lambda > lambdaMin);

        if verbose
            disp(['Decreasing lambda from ', num2str(lambda_old), ' to ', num2str(lambda)])
        end

        % accept changes
        U_old = u_bar; % save old for plotting
        u_bar              = unew(:, 1:N); % remove NaN
        xa0              = xnew;
        cost           = costnew;
        flgChange      = 1;



        % plotting
        if show_plots

            f1 = figure(1);
            if j == 1
                movegui(f1,'north');
                f1.Position = [5 471 1591 345];
            end

            if augment_states
                plot_columns = 6;
            else
                plot_columns = 5;
            end



            if u_lim_method == 1
                subplot(1, plot_columns, 1)
                stairs(t, unew')
                xlabel("Time")
                ylabel('Control action')
                legend('u1', 'u2')
            elseif u_lim_method == 2 && ~isempty(u_lims)
                subplot(1, plot_columns, 1)
                stairs(t, unew', '--')
                xlabel("Time")
                ylabel('"Pre-controls" and controls')
                u_squashed = (u_lims(:, 2) - u_lims(:, 1))/2.*tanh(unew) + (u_lims(:, 2) + u_lims(:, 1))/2;
                hold on
                stairs(t, u_squashed')
                ylim([0, 1])
                hold off
                legend('pre-u1', 'pre-u2', 'u1', 'u2')
            end


            % plot state space dynamics
            subplot(1, plot_columns, 2)
            plot(xnew(1, :), xnew(6, :), 'o-')
            xlabel('State 1 (Susceptible)')
            ylabel('State 6 (Theatened)')

            if ~isempty(Tracking_Trajectory)
                hold on
                plot(Tracking_Trajectory(1, :), Tracking_Trajectory(2, :), '*')
                hold off
            end

            % plot total cost
            costs_saved = [costs_saved, sum(costnew)];
            subplot(1, plot_columns, 3)
            plot(0:j, costs_saved)
            xlabel('Iteration')
            ylabel('Total Cost')

            % plot cost-to-go
            subplot(1, plot_columns, 4)
            plot(t, flip(cumsum(costnew(end:-1:1))))
            xlabel('Time')
            ylabel('Cost to go')

            % plot lambda
            lambdas_saved = [lambdas_saved, lambda];
            subplot(1, plot_columns, 5)
            plot(0:j, log(lambdas_saved))
            xlabel('Iteration')
            ylabel('Log(Lambda)')

            if augment_states
                % plot parameter estimate
                subplot(1, plot_columns, 6)
                plot(t, xnew(ip, :))
                xlabel('Time')
                ylabel('Parameter Estimates')
                %legend('p1', 'p2', 'p3')
            end

            drawnow();
        end

        % terminate ?
        if dcost < tolFun
            converged = 2;
            if verbose
                fprintf('\nSUCCESS: cost change < tolFun\n');
            end
            break;
        end

    else % no cost improvement
        % increase lambda
        lambda_old = lambda;

        dlambda  = max(dlambda * dlambda_0, dlambda_0);
        lambda   = max(lambda * dlambda, lambdaMin);

        if verbose
            disp(['NO STEP: iteration ', num2str(j)]);
            disp(['Increasing lambda from ', num2str(lambda_old), ' to ', num2str(lambda)])
        end

        costs_saved = [costs_saved, costs_saved(end)];
        lambdas_saved = [lambdas_saved, lambda];

        % terminate ?
        if lambda > lambdaMax

            if verbose
                disp('FINSIHED: lambda > lambdaMax')
            end

            % reset lambda for next outer loop iteration
            lambda = 1;
            dlambda = 1;

            converged = -1;
            warning('FINSIHED: lambda > lambdaMax')

            break;
        end

    end

    j = j + 1;

end

% Calculate dynamics noise covariance
% Pw = zeros(nxa, nxa, N);
% for k = N:-1:1
%     for i = 1:nw
%         C = c(:, i, k) + Cx(:, :, k, i)*(xnew(:, k) - xa_bar(:, k)) + Cu(:, :, k, i)*(unew(:, k) - u_bar(:, k));
%         Pw(:, :, k) = Pw(:, :, k) + C*C';
%     end
% end
%
% % Calculate measurement noise covariance
% Pv = zeros(ny, ny, N);
% for k = N:-1:1
%     for i = 1:nv
%         D = d(:, i, k) + Dx(:, :, k, i)*(xnew(:, k) - xa_bar(:, k)) + Du(:, :, k, i)*(unew(:, k) - u_bar(:, k));
%         Pv(:, :, k) = Pv(:, :, k) + D*D';
%     end
% end

if j == max_du_iterations
    converged = 3;
    if verbose
        disp('FINSIHED: Max iterations reached')

    end
end

end