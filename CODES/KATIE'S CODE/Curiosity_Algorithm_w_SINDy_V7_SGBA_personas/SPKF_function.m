% Sigma-point Kalman Filter for inverted pendulum with Combined Sigma Points (SPKF_pendulum_CSPs.m)
% Author: Andrew Mathis, andrew.mathis@unb.ca
% Based on the ECE5550 lecture notes by Dr. Gregory Plett
% Written Mar. 12, 2020

% Modification history:
% June 4, 2020: Changed sigma point generation using the CSPs such that error is reduced, see note below
% July 6, 2021: Added Idea #5 for CSP weights
% July 16, 2021: Added AdditiveNoise option to remove noise sigma points if the noise is known to be additive

function [x_hat_all, P_x] = SPKF_function(dt, Dynamics_F, Measurement_F, z_true_all, x_hat, w_hat, v_hat, P_x, P_w, P_v, h, AdditiveNoise)
Mode = 0;
% mode: 0 = SPKF, 1 = nested SPKF
% dt: time step
% x_true_all: vector of true states
% z_true_all: vector of measurements
% DiscreteStateDynamics: System dunamics function
% Measurement: Measurement function
% theta_true: true system parameters
% x_hat: initial state estimate
% w_hat: initial process noise estimate
% v_hat: initial measurement noise estimate
% P_x: initial state covariance
% P_w: process noise covariance
% P_v: measurement noise covariance
% h: CDKF scaling factor


%% Clear workspace
%clc
%clear
%close all
%rng(3) % fix random number generator for consistent results

%% Inputs
nx = size(x_hat, 1);
iterations = size(z_true_all, 1); % number of KF iterations to run
%iterations = iterations -1; % due to initial x_true

plotSigmaPointsDuringLoop = 0;
plotPendulumAnimation = 0;
delayTime = 0; % seconds

%x_true = x_true_all(1, :)'; % true initial value of the state vector (column vector)

nz = length(v_hat);

%% Setup
if AdditiveNoise == 1 % if noise is additive then there is no need to augment the state vector with the noise
    xa_hat = x_hat;
else
    xa_hat = [x_hat; w_hat; v_hat]; % augmented state vector to include noises
end

% Idea 0: baseline (how previously done) (MSE: 0.0830/0.0877, +/- 4.646)
if Mode == 0
    Wmx = [];
    L = length(xa_hat);
    Wmx(1) = (h^2 - L)/h^2; % calculate sigma point scaling factors for means and covariances
    Wmx(2:2*L+1) = 1/(2*h^2);
    Wcx = Wmx;
    nt = 0;
    N_points = 2*L + 1;

elseif Mode == 1
    nx = 1;
    nt = 1; % for CSPs, will have to fix later for joint estimation

    if AdditiveNoise == 1 % if noise is additive then there is no need to augment the state vector with the noise
        if nt > 0
            spaces = [nx, nt];
            nz = 0;
        else
            spaces = [nx];
            nz = 0;
        end
    else
        if nt > 0
            %N_spaces = 5;
            %spaces = [nx, nt, nx, nt, nz]; % equivalent to [nx, nt, n_wx, n_wt, n_v], was using until July 26, 2021
            spaces = [nx, nt, (nx + nt), nz];
        else
            %N_spaces = 3;
            spaces = [nx, nx, nz];
        end
    end

    N_spaces = length(spaces);

    CSPs = CombinationSigmaPointGenerator(nx, nt, nz, AdditiveNoise);

    L = (length(CSPs)-1)/2;

    Idea_Num = 4;

    switch Idea_Num
        case 5
            %% Calculate weights with Idea #5 from July 5, 2021
            % - nested points are just copies of normal SPKF state sigma points,
            %   so divide the original weights by the number of copies

            SP_W(1) = 1 - nx /(h^2); % center weight
            SP_W(2) = 1/(2*h^2); % edge weight

            num_nested_copies = length(find(sum(abs(CSPs(1:nx, :)), 1) == 0));
            Wmx(find(sum(abs(CSPs(1:nx, :)), 1) == 0 )) = SP_W(1) / num_nested_copies; % assign center weight divided by number of nested copies to all points with the states centered
            Wmx(find(sum(abs(CSPs(1:nx, :)), 1) ~= 0 )) = SP_W(2) / num_nested_copies;

            Wcx = Wmx;

        case 4
            %% calculate revised weights with idea 4

            combinations = fullfact(2*spaces+1)';

            Combined_sigma_points = [];
            Sigma_Point_weights = 1;
            for n = 1:length(spaces)
                L_space = spaces(n);

                % Generate the current space's sigma points
                Sigma_Points = [zeros(L_space, 1), eye(L_space), -1*eye(L_space)]

                % Use full factorial matrix to replicate current sigma points appropritaely for the nested combination
                Combined_sigma_points = [Combined_sigma_points; Sigma_Points(:, combinations(n, :))];

                L_old_weights = length(Sigma_Point_weights);

                % Replicate the old sigma point weights into the new space, and multiply the positions of its previous length by (1 - L/(h^2)) for
                % the new center points, and the other points by 1/(2 h^2) for the edge points.
                Sigma_Point_weights = repmat(Sigma_Point_weights, [1, 2*L_space+1]).*[(1 - L_space/(h^2)).*ones(1, L_old_weights), 1/(2*h^2).*ones(1, 2*L_space*L_old_weights)]
            end

            CSPs = Combined_sigma_points;
            L = (length(CSPs)-1)/2;
            Wmx = Sigma_Point_weights;
            Wcx = Wmx;
            N_points = 2*L + 1;



        case 3
            %% calculate revised weights with idea 3, generalized (MSE: 0.0830/0.0894, +/- ~5.839)
            % We = 1/(2*h^2);
            % c_coordins = find(sum(abs(CSPs(1:nx-nt, :))) == 0);
            % nc = length(c_coordins);
            % ne = 2*L+1 - nc;
            % Wc = (1 - We*ne)/nc;
            % Wmx(1:2*L+1) = We;
            % Wmx(c_coordins) = Wc; % overwrite We's with Wc as appropriate
            % Wcx = Wmx;

        case 1
            %% calculate revised weights with idea 1 % not generalized (MSE: 0.0439/0.0467, +/- 1.271)
            % Wmx(find(sum(abs(CSPs)) == 1)) = 1/(2*h^2);
            % Wmx(find(sum(abs(CSPs)) == 2)) = (1/(2*h^2))^2;
            % Wmx(find(sum(abs(CSPs)) == 3)) = (1/(2*h^2))^3;
            % Wmx(1) = 1 - sum(Wmx);
            % Wcx = Wmx;

        case 2
            %% calculate revised weights with idea 2 % not generalized (MSE: 0.0847/0.0903, +/- 5.453)
            % d = 1/(2*h^2);
            % c = 1 - 2^nz * d; % for v's
            % b = 1 - 2^nx * c; % for w's
            % a = 1 - 2^nx * b - (2^nx + 1) * 2^nx * c - (2^nx + 1) * (2^nx + 1) * 2^nz * d; % not quite correct
            % Wmx(find(sum(abs(CSPs)) == 1)) = b;
            % Wmx(find(sum(abs(CSPs)) == 2)) = c;
            % Wmx(find(sum(abs(CSPs)) == 3)) = d; % not generalized
            % Wmx(1) = 1 - sum(Wmx); %Wmx(1) = a;
            % Wcx = Wmx;
    end

    check_sum_to_one = sum(Wmx);
    %  if not(check_sum_to_one == 1)
    check_sum_to_one
    %    disp('Weights do not sum to one, simulation paused.')
    %    pause
    %end

    check_sum_to_zero = CSPs*Wmx';
    %  if check_sum_to_zero ~= 0
    check_sum_to_zero
    %     disp('Weighted CSPs do not sum to zero, simulation paused.')
    %      pause
    %  end


end

%x_true_all = zeros(iterations+1,length(x_true)); % preallocate vectors to save run data
%x_true_all(1,:) = x_true;
x_hat_all = zeros(iterations, length(x_hat));
P_x_all = zeros(iterations,length(x_hat)^2);
%z_true = Measurement(x_true, 0);
%z_true_all = zeros(iterations, length(z_true));


%% Kalman filter loop

for k = 1:iterations

    % Step 1a) State estimate prediction

    if AdditiveNoise == 1 % if noise is additive then there is no need to augment the state vector with the noise
        xa_hat = x_hat; % augmented state vector to include noises
        P_xa = blkdiag(P_x); % augmented covariance
    else
        xa_hat = [x_hat; w_hat; v_hat]; % augmented state vector to include noises
        if size(P_w, 3) ~= 1 || size(P_v, 3) ~= 1
            P_xa = blkdiag(P_x, P_w(:, :, k), P_v(:, :, k)); % added June 29, 2022 to account for noise with changing covariance
        else
            P_xa = blkdiag(P_x, P_w, P_v); % augmented covariance
        end
    end

    if ~isreal(P_xa)
        warning('Covariance matrix being adjusted to be real.')
        P_xa = real(P_xa);
    end

    if sum(eig(P_xa) <= 0) >= 1
        warning('Covariance matrix being adjusted to be positive definite.')
        P_xa = makePD(P_xa, 1e-15, 1);
    end

    if sum(eig(P_xa) <= 0) >= 1
        warning('Covariance matrix being adjusted to be positive definite.')
        P_xa = makePD(P_xa, 1e-5, 1);
    end

    if sum(eig(P_xa) <= 0) >= 1
        warning('Covariance matrix being adjusted to be positive definite.')
        P_xa = makePD(P_xa, 1e-1, 1);
    end

    if sum(eig(P_xa) <= 0) >= 1
        warning('Covariance matrix being adjusted to be positive definite.')
        P_xa = makePD(P_xa, 1, 1);
    end

    if ~isreal(P_xa)
        warning('Covariance matrix being adjusted to be real.')
        P_xa = real(P_xa);
    end

    Sigma_xa = chol(P_xa, 'lower'); % "square root" of augmented covariance

    % Use CSPs to form augmented sigma points:
    % - first method ignores off diagonal elements
    % - second method uses off diagonal elements, may have points in undesired locations, but has lower MSE error than first method in the one trial I've done so far

    %X_a_1 = xa_hat(:, ones([1, 2*L+1])) + h*diag(S_xa)*ones([1, 2*L+1]).*CSPs; % Augmented sigma points before nonlinear transformation
    %X_a_1 = xa_hat(:, ones([1, 2*L+1])) + h*S_xa*CSPs; % Augmented sigma points before nonlinear transformation

    %m_CSPS_1 = diag(S_xa)*ones([1, 2*L+1]).*CSPs;
    %m_CSPS_2 = S_xa*CSPs;

    if Mode == 0
        X_a_1 = xa_hat(:, ones([1, 2*L+1])) + h*[zeros([L, 1]), Sigma_xa, -Sigma_xa]; % Augmented sigma points before nonlinear transformation



    elseif Mode == 1
        %X_a_1 = xa_hat(:, ones([1, 2*L+1])) + h*diag(S_xa)*ones([1, 2*L+1]).*CSPs; % Augmented sigma points before nonlinear transformation
        X_a_1 = xa_hat(:, ones([1, 2*L+1])) + h*Sigma_xa*CSPs; % Augmented sigma points before nonlinear transformation
    end

    if AdditiveNoise == 1 % if noise is additive then there is no need to augment the state vector with the noise
        X_x_1 = X_a_1(1:nx+nt, :); % State sigma points before nonlinear transformation

        % Simulate process with randomly generated noise INSTEAD of using noise sigma points, as noise is additive
        sqrtP_w = chol(P_w, 'lower');
        process_noise = sqrtP_w* randn(size(X_x_1));
        % OR
        %process_noise = zeros(size(X_x_1));

        X_x_2 = Dynamics_F(dt, X_x_1, process_noise); % State sigma points after nonlinear transformation, f(x, u, w) hard coded for efficiency
    else
        X_x_1 = X_a_1(1:nx+nt, :); % State sigma points before nonlinear transformation
        X_w_1 = X_a_1(nx+nt+1:2*(nx+nt), :); % Process noise sigma points before nonlinear transformation
        X_v_1 = X_a_1(2*(nx+nt)+1:end, :); % Measurement noise points before nonlinear transformation
        X_x_2 = Dynamics_F(dt, X_x_1, X_w_1); % State sigma points after nonlinear transformation, f(x, u, w) hard coded for efficiency
    end

    x_hat = X_x_2 * Wmx'; % state estimate prediction
    save_xhat = x_hat;

    % Step 1b) State error covariance predictions
    X_diff_first = X_x_2(:,1) - x_hat;
    X_diff_rest = X_x_2(:,2:end) - x_hat(:,ones([1 N_points-1]));
    P_x = Wcx(1)*(X_diff_first*X_diff_first') + Wcx(2:end).*X_diff_rest*X_diff_rest'; % state error covariance prediction
    save_Px = P_x;

    % Step 1c) Output prediction
    if AdditiveNoise == 1 % if noise is additive then there is no need to augment the state vector with the noise

        % Simulate process with randomly generated noise INSTEAD of using noise sigma points, as noise is additive
        sqrtP_v = chol(P_v, 'lower');
        measurement_noise = sqrtP_v* randn(size(X_x_2));
        % OR
        %measurement_noise = zeros(size(X_x_2));

        Z = Measurement_F(X_x_2, measurement_noise); % measurement sigma points, h(x, u, v)
    else
        Z = Measurement_F(X_x_2, X_v_1); % measurement sigma points, h(x, u, v)
    end

    z_hat = Z * Wmx'; % measurement prediction

    % Simulate true system and get measurement
    %v = chol(Sigma_v)'*randn([nz, 1]);
    %z_true = Measurement(x_true, v); % z is based on present x and u, h(x, u, v)
    z_true = z_true_all(k, :)';

    %w = chol(Sigma_w)'*randn([nx, 1]);
    %x_true = DiscreteStateDynamics(k, dt, x_true, theta_true, w); % future x is based on present u, f(x, u, w)
    %x_true = x_true_all(k+1, :)';

    % Step 2a) Calculate Kalman gain matrix
    Z_diff_first = Z(:,1) - z_hat;
    Z_diff_rest = Z(:,2:end) - z_hat(:,ones([1 N_points-1])); % sqrt here as will square in a second
    Sigma_z = Wcx(1)*(Z_diff_first*Z_diff_first') + Wcx(2:end).*Z_diff_rest*Z_diff_rest';
    Sigma_xz = Wcx(1)*(X_diff_first*Z_diff_first') + Wcx(2:end).*X_diff_rest*Z_diff_rest';
    K_x = Sigma_xz/Sigma_z;

    % Step 2b) State estimate update
    x_hat = x_hat + K_x*(z_true - z_hat); % update state prediction

    % Step 2c) State error covariance update
    P_x = P_x - K_x*Sigma_z*K_x';

    % Save run data
    %x_true_all(k+1,:) = x_true;
    x_hat_all(k,:) = x_hat;
    P_x_all(k,:) = P_x(:);
    %z_true_all(k+1,:) = z_true;

    Xs_before(:,:, k) = X_x_1; % sigma points before NL transform
    x_hat_before(k,:) = save_xhat; % x_hat before update()
    Px_before(k,:) = save_Px(:); % P_x before update()
    Xs_after(:,:, k) = X_x_2; % sigma points after NL transform

    % Plot Sigma points
    %     if plotSigmaPointsDuringLoop == 1
    %         xa_hat = [x_hat; w_hat; v_hat]; % augmented state vector to include noises
    %         P_xa = blkdiag(P_x, P_w, P_v); % augmented covariance
    %         Sigma_xa = chol(P_xa, 'lower'); % "square root" of augmented covariance
    %
    %         X_a_3 = xa_hat(:, ones([1, 2*L+1])) + h*diag(Sigma_xa)*ones([1, 2*L+1]).*CSPs; % Augmented sigma points before nonlinear transformation
    %         X_x_3 = X_a_3(1:nx, :); % State sigma points before nonlinear transformation
    %
    %         colour = rand(1,3);
    %         plot(X_x_1(1, :), X_x_1(2, :), 'kx', 'DisplayName', 'Initial')
    %         hold on
    %         plot(X_x_2(1, :), X_x_2(2, :), 'r*', 'DisplayName', 'Predicted')
    %         plot(x_true_all(k,1), x_true_all(k,2), 'go', 'DisplayName', 'True states')
    %         plot(X_x_3(1, :), X_x_3(2, :), 'bs', 'DisplayName', 'Updated')
    %         %plot(z_true(1), z_true(2), 'bo', 'DisplayName', 'Measurement')
    %
    %         hold off
    %         legend
    %         xlabel('State 1');
    %         ylabel('State 2');
    %         title('Sigma-point propogation');
    %         xlim([-3, 3])
    %         ylim([-3, 3])
    %         pause(delayTime)
    %     end
end

% %% Plot run data
% figure;
% for i = 1:nx
%     colour(i, :) = rand(1,3);
%     hold on
%     plot(0:iterations-1, x_true_all(1:iterations, i), '-', 'DisplayName', ['true x_', num2str(i)], 'color', colour(i, :));
%     plot(0:iterations-1, x_hat_all(:, i),'--', 'DisplayName', ['estimate x_', num2str(i)], 'color', colour(i, :));
%     plot(0:iterations-1, x_hat_all(:, i)+3*sqrt(Sigma_x_all(:, nx*(i-1)+i)),'-.', 'DisplayName', ['bound x_', num2str(i)], 'color', colour(i, :));
%     plot(0:iterations-1, x_hat_all(:, i)-3*sqrt(Sigma_x_all(:, nx*(i-1)+i)),'-.', 'color', colour(i, :),'HandleVisibility','off');
%     plot(0:iterations-1, z_true_all(1:iterations, i), 'o', 'DisplayName', ['true z_', num2str(i)], 'color', colour(i, :));
% end
% grid;
% legend;
% xlabel('Iteration');
% ylabel('State');
% title('Sigma-point Kalman filter');
%
% figure;
% for i = 1:nx
%     hold on
%     MSE(i) = mean((x_true_all(1:iterations, i) - x_hat_all(:, i)).^2);
%     plot(0:iterations-1, x_true_all(1:iterations, i) - x_hat_all(:, i), '-', 'DisplayName', ['error x_', num2str(i), ' (MSE: ', num2str(MSE(i)), ')'], 'color', colour(i, :));
%     plot(0:iterations-1, 3*sqrt(Sigma_x_all(:, nx*(i-1)+i)), '--', 'DisplayName', ['bounds x_', num2str(i)], 'color', colour(i, :));
%     plot(0:iterations-1,-3*sqrt(Sigma_x_all(:, nx*(i-1)+i)), '--', 'color', colour(i, :),'HandleVisibility','off');
% end
% grid;
% legend;
% title('SPKF State error with bounds');
% xlabel('Iteration');
% ylabel('Estimation Error');
% %ylim([-2 2])
%
% %% Plot system dynamics
%
% if plotPendulumAnimation
%     pause(1);
%     plotPendulum(x_true_all, ones(1, length(x_true_all)), 5, dt*(iterations-1))
% end

%% State dynamics and measurement functions

% function xk = DiscreteStateDynamics(k, dt, xkminus, theta, w) % also function of ukminus
%
% %xk = xkminus + StateDynamics(t, xkminus, theta)*dt + w;
%
% [r, points] = size(xkminus);
%
% for p = 1:points
%
%     [t_sim, x_sim] = ode45(@(t, x)StateDynamics(t, xkminus(:, p), theta), [(k-1)*dt, k*dt], xkminus(:, p));
%
%     xk(:, p) = x_sim(end, :)';
%
% end
%
% xk = xk + w;
%
% end
%
% function dxdt = StateDynamics(t, x, theta)
%
% % ---- dynamic constraints --------
% % System properties
% % m = 1; % kg
% % l = 5; % m
% % b = 0.25; % Nm
% g = 9.81; % m/s2
%
% m = theta(1); % kg
% l = theta(2); % m
% b = theta(3); % Nm
%
%
% % "greedy control"
% u_max = 50; % Nm , possible at 36 Nm
%
% if x(1) >= 0
%     u = u_max;
% elseif x(1) < 0
%     u = -u_max;
% end
%
% %u = u_max;
%
% %*** angles defined from the vertical! ***
% dxdt = [x(2); (-u + b*x(2) + m*g*l*sin(x(1)))/(m*l^2)]; % dx/dt = f(x,u)
%
% end
%
% function z = Measurement(x, v) % also function of uk, theta
%
% z = x + v;
%
% end
end