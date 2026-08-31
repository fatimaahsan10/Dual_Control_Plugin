%% ContinuousStateDynamics.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

function [xdot, nx, np, F, Fp] = ContinuousStateDynamics(dt, xa, u, c, augment_states, u_lims, u_lim_method)

Nplus1 = size(xa, 2);

if u_lim_method == 2 && ~isempty(u_lims)% squashing method
    u = (u_lims(:, 2) - u_lims(:, 1))/2.*tanh(u) + (u_lims(:, 2) + u_lims(:, 1))/2;
end

% unpack controls:
u1 = u(1, :);
u2 = u(2, :);
u3 = u(3, :);
u4 = u(4, :);
u5 = u(5, :);
%u6 = u(6, :);
%u7 = u(7, :);

% unpack states:
%xa = xa./c(1);

S = xa(1, :);
I = xa(2, :);
D = xa(3, :);
A = xa(4, :);
R = xa(5, :);
T = xa(6, :);
H = xa(7, :);
E = xa(8, :);

nx = 8;

if augment_states % if states have been augmented with parameters, unpack parameters. Also unpack constants.
    % parameters unpacked from xa(#, :) after states
    alpha_min = xa(9, :);
    alpha_max = xa(10, :);
    eta_alpha_1 = xa(11, :);

    beta_min = xa(12, :);
    beta_max = xa(13, :);
    eta_beta_1 = xa(14, :);
    eta_beta_2 = xa(15, :);

    gamma_min = xa(16, :);
    gamma_max = xa(17, :);
    eta_gamma_1 = xa(18, :);

    epsilon_min = xa(19, :);
    epsilon_max = xa(20, :);
    eta_epsilon_1 = xa(21, :);

    theta_min = xa(22, :);
    theta_max = xa(23, :);
    eta_theta_1 = xa(24, :);

%     sigma1_min = xa(25, :);
%     sigma1_max = xa(26, :);

%     sigma2_min = xa(27, :);
%     sigma2_max = xa(28, :);

%     tau1_min = xa(29, :);
%     tau1_max = xa(30, :);

%     tau2_min = xa(31, :);
%     tau2_max = xa(32, :);

%     T_ICU_min = xa(33, :);
%     T_ICU_max = xa(34, :);

    np = 16; % number of parameters above

    % unpack constants:

    N = c(1);
    zeta = c(2);
    lambda = c(3);
    kappa = c(4);
    mu_1 = c(5);
    mu_2 = c(6);
    tau_crit = c(7);
    sigma_1 = c(8);
    sigma_2 = c(9);
    tau_1 = c(10);
    tau_2 = c(11);
    T_ICU = c(12);

else % adaptive case
    % Note: in this case c = [constants; p_hat];
    % unpack constants:
    N = c(1);
    zeta = c(2);
    lambda = c(3);
    kappa = c(4);
    mu_1 = c(5);
    mu_2 = c(6);
    tau_crit = c(7);
    sigma_1 = c(8);
    sigma_2 = c(9);
    tau_1 = c(10);
    tau_2 = c(11);
    T_ICU = c(12);

    % unpack unaugmented parameters fro constants vector after true constants:

    alpha_min = c(13);
    alpha_max = c(14);
    eta_alpha_1 = c(15);

    beta_min = c(16);
    beta_max = c(17);
    eta_beta_1 = c(18);
    eta_beta_2 = c(19);

    gamma_min = c(20);
    gamma_max = c(21);
    eta_gamma_1 = c(22);

    epsilon_min = c(23);
    epsilon_max = c(24);
    eta_epsilon_1 = c(25);

    theta_min = c(26);
    theta_max = c(27);
    eta_theta_1 = c(28);

    np = 0; % all treated at constants, so no parameters (do not modify)
end

%% Dynamic model

% algorithmic variables
mu = mu_1 + mu_2;

eta_alpha_2 = 1 - eta_alpha_1;
eta_beta_5 = 1 - eta_beta_1 - eta_beta_2;
eta_gamma_2 = 1 - eta_gamma_1;
eta_epsilon_3 = 1 - eta_epsilon_1;
eta_theta_4 = 1 - eta_theta_1;

% Kohler 2021 original:
% alpha = alpha_max + (alpha_min - alpha_max).*u1;
% gamma = gamma_max + (gamma_min - gamma_max).*u1;

% ACM modification:
alpha = alpha_max + (alpha_min - alpha_max).*(u1 .* eta_alpha_1 + u2 .* eta_alpha_2);
beta = beta_max + (beta_min - beta_max).*(u1 .* eta_beta_1 + u2 .* eta_beta_2 + u5 .* eta_beta_5);
gamma = gamma_max + (gamma_min - gamma_max).*(u1 .* eta_gamma_1 + u2 .* eta_gamma_2);
%tau1 = tau1_max + (tau1_min - tau1_max).*u6;
%tau2 = tau2_max + (tau2_min - tau2_max).*u7;

epsilon = epsilon_min + (epsilon_max - epsilon_min).*(u1 .* eta_epsilon_1 + u3 .* eta_epsilon_3);
theta = theta_min + (theta_max - theta_min).*(u1 .* eta_theta_1 + u4 .* eta_theta_4);
%sigma1 = sigma1_min + (sigma1_max - sigma1_min).*u6;
%sigma2 = sigma2_min + (sigma2_max - sigma2_min).*u7;
%T_ICU = T_ICU_min + (T_ICU_max - T_ICU_min).*u7;

% percentage states:
tauT = mu_1./mu.*tau_1.*T + max(mu_2./mu.*tau_2.*T, tau_2.*T_ICU + tau_crit.*(mu_2./mu.*T - T_ICU));
sigmaT = mu_1./mu.*sigma_1.*T + sigma_2.*min(mu_2/mu.*T, T_ICU);

% dx/dt = f(x,u), use elementwise operations
f = [-S./N.*(alpha.*I + beta.*D + gamma.*A + beta.*R); ... % Susceptable
    S./N.*(alpha.*I + beta.*D + gamma.*A + beta.*R) - (epsilon + zeta + lambda).*I; ... % Infected
    epsilon.*I - (zeta + lambda).*D; ... % Diagnosed 
    zeta.*I - (theta + mu + kappa).*A; ... % Ailing
    zeta.*D + theta.*A - (mu + kappa).*R; ... % Recognized
    mu.*A + mu.*R - (sigmaT + tauT); ... % Threatened
    lambda.*I + lambda.*D + kappa.*A + kappa.*R + sigmaT; ... % Healed
    tauT]; % Extinct
    % 



% % state constraints
S = max(S, 0);
I = max(I, 0);
D = max(D, 0);
A = max(A, 0);
R = max(R, 0);
T = max(T, 0);
H = max(H, 0);
E = max(E, 0);

S = min(S, c(1));
I = min(I, c(1));
D = min(D, c(1));
A = min(A, c(1));
R = min(R, c(1));
T = min(T, c(1));
H = min(H, c(1));
E = min(E, c(1));



if S <= 0 
    f(1) = max(f(1), 0);
elseif S >= c(1)
    f(1) = min(f(1), 0);
end

if I <= 0
    f(2) = max(f(2), 0);
elseif I >= c(1)
    f(2) = min(f(2), 0);
end

if D <= 0
    f(3) = max(f(3), 0);
elseif D >= c(1)
    f(3) = min(f(3), 0);
end

if A <= 0
    f(4) = max(f(4), 0);
elseif A >= c(1)
    f(4) = min(f(4), 0);
end

if R <= 0
    f(5) = max(f(5), 0);
elseif R >= c(1)
    f(5) = min(f(5), 0);
end

if T <= 0
    f(6) = max(f(6), 0);
elseif T >= c(1)
    f(6) = min(f(6), 0);
end

if H <= 0
    f(7) = max(f(7), 0);
elseif H >= c(1)
    f(7) = min(f(7), 0);
end

if E <= 0
    f(8) = max(f(8), 0);
elseif E >= c(1)
    f(8) = min(f(8), 0);
end

%% Noise

if augment_states
    fp = 0.*ones(np, Nplus1); % parameter dynamics, before noise (modify this line)
else
    fp = [];
end
fa = [f; fp];
xdot = fa;

%% Noise

F = 1e-5.*eye(nx); % dynamics noise, [nx by nw] (modify this line)
if augment_states
    Fp = 1e-15.*ones(np); % parameter noise, [np by nw] (modify this line)
else
    Fp = [];
end


end