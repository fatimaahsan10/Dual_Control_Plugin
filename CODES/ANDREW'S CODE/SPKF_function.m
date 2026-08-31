%% Sigma-point Kalman Filter
% Author: Andrew Mathis, andrew.mathis@unb.ca
% Written Mar. 12, 2020

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

plotSigmaPointsDuringLoop = 0;
plotPendulumAnimation = 0;
delayTime = 0; % seconds

nz = length(v_hat);

%% Setup
if AdditiveNoise == 1
    xa_hat = x_hat;
else
    xa_hat = [x_hat; w_hat; v_hat]; % augmented state vector to include noises
end

Wmx = [];
L = length(xa_hat);
Wmx(1) = (h^2 - L)/h^2; % calculate sigma point scaling factors for means and covariances
Wmx(2:2*L+1) = 1/(2*h^2);
Wcx = Wmx;
nt = 0;
N_points = 2*L + 1;


x_hat_all = zeros(iterations, length(x_hat));
P_x_all = zeros(iterations,length(x_hat)^2);



%% Kalman filter loop

for k = 1:iterations

    % Step 1a) State estimate prediction

    if AdditiveNoise == 1 % if noise is additive then there is no need to augment the state vector with the noise
        xa_hat = x_hat; % augmented state vector to include noises
        P_xa = blkdiag(P_x); % augmented covariance
    else
        xa_hat = [x_hat; w_hat; v_hat]; % augmented state vector to include noises
        if size(P_w, 3) ~= 1 || size(P_v, 3) ~= 1
            P_xa = blkdiag(P_x, P_w(:, :, k), P_v(:, :, k));
        else
            P_xa = blkdiag(P_x, P_w, P_v); % augmented covariance
        end
    end

    if ~isreal(P_xa)
        warning('Covariance matrix being adjusted to be real.')
        P_xa = real(P_xa);
    end

    P_xa = real(P_xa);

    if sum(eig(P_xa) <= 0) >= 1
        %warning('Covariance matrix being adjusted to be positive definite.')
        P_xa = makePD(P_xa, 1e-5, 1);
    end

    Sigma_xa = chol(P_xa, 'lower'); % "square root" of augmented covariance


    X_a_1 = xa_hat(:, ones([1, 2*L+1])) + h*[zeros([L, 1]), Sigma_xa, -Sigma_xa]; % Augmented sigma points before nonlinear transformation


    if AdditiveNoise == 1 % if noise is additive then there is no need to augment the state vector with the noise
        X_x_1 = X_a_1(1:nx+nt, :); % State sigma points before nonlinear transformation

        % Simulate process with randomly generated noise
        sqrtP_w = chol(P_w, 'lower');
        process_noise = sqrtP_w* randn(size(X_x_1));


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

        % Simulate process with randomly generated noise
        sqrtP_v = chol(P_v, 'lower');
        measurement_noise = sqrtP_v* randn(size(X_x_2));


        Z = Measurement_F(X_x_2, measurement_noise); % measurement sigma points, h(x, u, v)
    else
        Z = Measurement_F(X_x_2, X_v_1); % measurement sigma points, h(x, u, v)
    end

    z_hat = Z * Wmx'; % measurement prediction

    % Simulate true system and get measurement
    z_true = z_true_all(k, :)';

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
    x_hat_all(k,:) = x_hat;
    P_x_all(k,:) = P_x(:);


    Xs_before(:,:, k) = X_x_1; % sigma points before NL transform
    x_hat_before(k,:) = save_xhat; % x_hat before update()
    Px_before(k,:) = save_Px(:); % P_x before update()
    Xs_after(:,:, k) = X_x_2; % sigma points after NL transform

end


end