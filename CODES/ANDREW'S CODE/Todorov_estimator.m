%% Todorov_estimator.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

% Based on:
% - Li, W., & Todorov, E. (2007). Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system. International Journal of Control, 80(9), 1439–1453. https://doi.org/10.1080/00207170701364913

function [x_hat, pi, K, m_x, m_e, Sigma_x, Sigma_e, Sigma_xe, Sigma_ex]  = Todorov_estimator(A, B, c, Cx, Cu, d, Dx, Du, E, F, l, L, x_bar, y_bar, u_bar, x_hat_0, cov_x_hat_0, dt, constants, sqrtR, augment_states)

    % Note: x_hat here is the estimate of x, where x is the DEVIATION variable

    % dimensional variables
    nx = size(x_hat_0, 1);
    ny = size(y_bar, 1);
    nu = size(Cu, 2);
    nw = size(c, 2);
    nv = size(d, 2);
    N = size(c, 3) - 1;

    % preallocate all variables:
    x_hat = zeros(nx, N+1);
    m_x = zeros(nx, N+1);
    m_e = zeros(nx, N+1);
    Sigma_x = zeros(nx, nx, N+1);
    Sigma_e = zeros(nx, nx, N+1);
    Sigma_xe = zeros(nx, nx, N+1);
    Sigma_ex = zeros(nx, nx, N+1);
    P = zeros(ny, ny, N); 
    M = zeros(nx, nx, N);
    K = zeros(nx, ny, N);
    pi = zeros(nu, N);
    y = zeros(ny, N+1);

    % initialize variables
    x_hat(:, 1) = x_hat_0;
    m_x(:, 1) = x_hat_0;
    Sigma_x(:, :, 1) = x_hat(:, 1)*x_hat(:, 1)';
    Sigma_e(:, :, 1) = cov_x_hat_0;
    Sigma_xe(:, :, 1) = zeros(size(cov_x_hat_0));
    Sigma_ex(:, :, 1) = Sigma_xe(:, :, 1)'; 

for k = 1:N
    
    % Shorthand for P and M calcs
    mxme = m_x(:, k) + m_e(:, k);
    lLmx = l(:,k) + L(:, :, k)*m_x(:, k);
    Sigma_xSigma_ex = Sigma_x(:, :, k) + Sigma_ex(:, :, k);
    Sigma_xSigma_xe = Sigma_x(:, :, k) + Sigma_xe(:, :, k);
    All_Sigmas = Sigma_x(:, :, k) + Sigma_xe(:, :, k) + Sigma_ex(:, :, k) + Sigma_e(:, :, k);
    l_UU_terms = l(:,k)*l(:,k)' + l(:, k)*m_x(:, k)'*L(:, :, k)' + L(:, :, k)*m_x(:, k)*l(:, k)' + L(:, :, k)*Sigma_x(:, :, k)*L(:, :, k)';

    for j = 1:nv
        P(:, :, k) = P(:, :, k) + d(:, j, k)*d(:, j, k)' + d(:, j, k)*mxme'*Dx(:, :, k, j)' ... % Todorov 2007, EQN 56
            + Dx(:, :, k, j)*mxme*d(:, j, k)' + d(:, j, k)*lLmx'*Du(:, :, k, j)' ...
            + Du(:, :, k, j)*lLmx*d(:, j, k)' ...
            + Dx(:, :, k, j)*(mxme*l(:, k)' + Sigma_xSigma_ex*L(:, :, k)')*Du(:, :, k, j)' ...
            + Du(:, :, k, j)*(l(:, k)*mxme' + L(:, :, k)*Sigma_xSigma_xe)*Dx(:, :, k, j)' ...
            + Dx(:, :, k, j)*All_Sigmas*Dx(:, :, k, j)' ...
            + Du(:, :, k, j)*l_UU_terms*Du(:, :, k, j)';
    end

    for i = 1:nw
        M(:, :, k) = M(:, :, k) + c(:, i, k)*c(:, i, k)' + c(:, i, k)*mxme'*Cx(:, :, k, i)' ... % Todorov 2007, EQN 57
            + Cx(:, :, k, i)*mxme*c(:, i, k)' + c(:, i, k)*lLmx'*Cu(:, :, k, i)' ...
            + Cu(:, :, k, i)*lLmx*c(:, i, k)' ...
            + Cx(:, :, k, i)*(mxme*l(:, k)' + Sigma_xSigma_ex*L(:, :, k)')*Cu(:, :, k, i)' ...
            + Cu(:, :, k, i)*(l(:, k)*mxme' + L(:, :, k)*Sigma_xSigma_xe)*Cx(:, :, k, i)' ...
            + Cx(:, :, k, i)*All_Sigmas*Cx(:, :, k, i)' ...
            + Cu(:, :, k, i)*l_UU_terms*Cu(:, :, k, i)';
    end

    % Calculate estimator gain and control action
    K(:, :, k) = A(:, :, k)*Sigma_e(:, :, k)*F(:, :, k)'/(F(:, :, k)*Sigma_e(:, :, k)*F(:, :, k)' + P(:, :, k)); % Todorov 2007, EQN 50
    
    pi(:, k) = l(:, k) + L(:, :, k)*x_hat(:, k);

    % Get y
    x_data = x_bar(:, k) + x_hat(:, k);

    vs = 0.*sqrtR*randn([nv, 1]); 

    y_data(:, k)  = Measurement(dt, x_data, u_bar(:, k) + pi(:, k), constants, vs, false, augment_states, 0);
    y(:, k) = y_data(:, k) - y_bar(:, k);

    % Calculate next state estimate
    x_hat(:, k+1) = A(:, :, k)*x_hat(:, k) + B(:, :, k)*pi(:, k) + K(:, :, k)*(y(:, k) - F(:, :, k)*x_hat(:, k) - E(:, :, k)*pi(:, k));  % Todorov 2007, EQN 49

    % Shorthand for m and Sigma calcs
    ABL = A(:, :, k) + B(:, :, k)*L(:, :, k);
    KF = K(:, :, k)*F(:, :, k);
    AKF = A(:, :, k) - KF;

    % Calculate next unconditional means and covariances
    m_x(:, k+1) = ABL*m_x(:, k) + KF*m_e(:, k) + B(:, :, k)*l(:, k);   % Todorov 2007, EQN 51
    m_e(:, k+1) = AKF*m_e(:, k);  % Todorov 2007, EQN 52

    Sigma_x(:, :, k+1) = ABL*Sigma_x(:, :, k)*ABL' + KF*Sigma_e(:, :, k)*A(:, :, k)' ...  % Todorov 2007, EQN 53
        + ABL*Sigma_xe(:, :, k)*F(:, :, k)'*K(:, :, k)' + KF*Sigma_ex(:, :, k)*ABL' ...
        + (ABL*m_x(:, k) + KF*m_e(:, k))*l(:, k)'*B(:, :, k)' ...
        + B(:, :, k)*l(:, k)*(ABL*m_x(:, k) + KF*m_e(:, k))' + B(:, :, k)*l(:, k)*l(:, k)'*B(:, :, k)';

    Sigma_e(:, :, k+1) = AKF*Sigma_e(:, :, k)*A(:, :, k)' + M(:, :, k);  % Todorov 2007, EQN 54

    Sigma_xe(:, :, k+1) = ABL*Sigma_xe(:, :, k)*AKF' + B(:, :, k)*l(:, k)*m_e(:, k)'*AKF';     % Todorov 2007, EQN 55

    Sigma_ex(:, :, k+1) = Sigma_xe(:, :, k+1)'; 
end


end

%% P and M calcs without shorthand
%     for j = 1:nv
%         P(:, k) = P(:, k) + d(:, j, k)*d(:, j, k)' + d(:, j, k)*(m_x(:, k) + m_e(:, k))'*Dx(:, :, k, j)' ...
%             + Dx(:, :, k, j)*(m_x(:, k) + m_e(:, k))*d(:, j, k)' + d(:, j, k)*(l(k) + L(:, :, k)*m_x(:, k))'*Du(:, :, k, j)' ...
%             + Du(:, :, k, j)*(l(k) + L(:, :, k)*m_x(:, k))*d(:, j, k)' ...
%             + Dx(:, :, k, j)*((m_x(:, k) + m_e(:, k))*l(k)' + (Sigma_x(:, k) + Sigma_ex(:, k))*L(:, :, k)')*Du(:, :, k, j)' ...
%             + Du(:, :, k, j)*(l(k)*(m_x(:, k) + m_e(:, k))' + L(:, :, k)*(Sigma_x(:, k) + Sigma_xe(:, k)))*Dx(:, :, k, j)' ...
%             + Dx(:, :, k, j)*(Sigma_x(:, k) + Sigma_xe(:, k) + Sigma_ex(:, k) + Sigma_e(:, k))*Dx(:, :, k, j)' ...
%             + Du(:, :, k, j)*(l(k)*l(k)' + l(k)*m_x(:, k)'*L(:, :, k)' + L(:, :, k)*m_x(:, k)*l(k)' + L(:, :, k)*Sigma_x(:, k)*L(:, :, k)')*Du(:, :, k, j)';
%     end
% 
%     for i = 1:nw
%         M(:, k) = M(:, k) + c(:, i, k)*c(:, i, k)' + c(:, i, k)*(m_x(:, k) + m_e(:, k))'*Cx(:, :, k, i)' ...
%             + Cx(:, :, k, i)*(m_x(:, k) + m_e(:, k))*c(:, i, k)' + c(:, i, k)*(l(k) + L(:, :, k)*m_x(:, k))'*Cu(:, :, k, i)' ...
%             + Cu(:, :, k, i)*(l(k) + L(:, :, k)*m_x(:, k))*c(:, i, k)' ...
%             + Cx(:, :, k, i)*((m_x(:, k) + m_e(:, k))*l(k)' + (Sigma_x(:, k) + Sigma_ex(:, k))*L(:, :, k)')*Cu(:, :, k, i)' ...
%             + Cu(:, :, k, i)*(l(k)*(m_x(:, k) + m_e(:, k))' + L(:, :, k)*(Sigma_x(:, k) + Sigma_xe(:, k)))*Cx(:, :, k, i)' ...
%             + Cx(:, :, k, i)*(Sigma_x(:, k) + Sigma_xe(:, k) + Sigma_ex(:, k) + Sigma_e(:, k))*Cx(:, :, k, i)' ...
%             + Cu(:, :, k, i)*(l(k)*l(k)' + l(k)*m_x(:, k)'*L(:, :, k)' + L(:, :, k)*m_x(:, k)*l(k)' + L(:, :, k)*Sigma_x(:, k)*L(:, :, k)')*Cu(:, :, k, i)';
%     end