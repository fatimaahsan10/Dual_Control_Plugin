%% backward_pass.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

% Based on Li and Todorov (2007): Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system

function [diverge, l, L, s0_alpha] = backward_pass(A, B, c, Cx, Cu, d, Dx, Du, E, F, K, fxx, fxu, fuu, q0, q, Q, r, R, P,lambda,regType,lims,u, nv)

vectens = @(a,b) permute(sum(bsxfun(@times,a,b),1), [3 2 1]);

% Size variables
N  = size(q,2);
n  = numel(q)/N;
m  = numel(r)/N;
nw = size(c, 2);
%nv = size(d, 1);

% Reshape / preallocate matricies
q    = reshape(q,  [n N]);
r    = reshape(r,  [m N]);
Q   = reshape(Q, [n n N]);
P   = reshape(P, [n m N]);
R   = reshape(R, [m m N]);

l     = zeros(m,N-1);
L     = zeros(m,n,N-1);

sx    = zeros(n,N);
sxh    = zeros(n,N);

Sx   = zeros(n,n,N);
Sxh   = zeros(n,n,N);
Sxxh   = zeros(n,n,N);

% Set initial values
s0_alpha = [0 0];

Sx(:,:,N)   = Q(:,:,N);             % part of EQN (17) in Todorov 2007
Sxh(:,:,N)  = zeros([n, n, 1]);     % part of EQN (18) in Todorov 2007
Sxxh(:,:,N) = zeros([n, n, 1]);     % part of EQN (19) in Todorov 2007

sx(:,N)     = q(:,N);               % part of EQN (20) in Todorov 2007
sxh(:,N)    = zeros([n, 1]);        % part of EQN (21) in Todorov 2007

s0(N)       = q0(N);                % part of EQN (22) in Todorov 2007


% Start backward pass
diverge  = 0;
for k = N-1:-1:1

    % Pre-calculate C and D noise sums 
    c_Sx_c =   zeros(1);
    Cx_Sx_c =  zeros(n,1);
    Cx_Sx_Cx = zeros(n,n);
    Cu_Sx_c =  zeros(m,1);
    Cu_Sx_Cx = zeros(m,n);
    Cu_Sx_Cu = zeros(m,m);

    for i = 1:nw
        c_Sx_c =   c_Sx_c     + c(:, i, k)'*Sx(:,:,k+1)*c(:, i, k);            % s0
        Cx_Sx_c =  Cx_Sx_c    + Cx(:, :, k, i)'*Sx(:,:,k+1)*c(:, i, k);        % s
        Cx_Sx_Cx = Cx_Sx_Cx   + Cx(:, :, k, i)'*Sx(:,:,k+1)*Cx(:, :, k, i);    % S
        Cu_Sx_c =  Cu_Sx_c    + Cu(:, :, k, i)'*Sx(:,:,k+1)*c(:, i, k);        % g
        Cu_Sx_Cx = Cu_Sx_Cx   + Cu(:, :, k, i)'*Sx(:,:,k+1)*Cx(:, :, k, i);    % G
        Cu_Sx_Cu = Cu_Sx_Cu   + Cu(:, :, k, i)'*Sx(:,:,k+1)*Cu(:, :, k, i);    % H
    end

    d_K_Sxh_K_d =   zeros(1);
    Dx_K_Sxh_K_d =  zeros(n,1);
    Dx_K_Sxh_K_Dx = zeros(n,n);
    Du_K_Sxh_K_d =  zeros(m,1);
    Du_K_Sxh_K_Dx = zeros(m,n);
    Du_K_Sxh_K_Du = zeros(m,m);

    for i = 1:nv
        d_K_Sxh_K_d = d_K_Sxh_K_d + d(:, i, k)'*K(:, :, k)'*Sxh(:,:,k+1)*K(:, :, k)*d(:, i, k); % s0
        Dx_K_Sxh_K_d = Dx_K_Sxh_K_d + Dx(:, :, k, i)'*K(:, :, k)'*Sxh(:,:,k+1)*K(:, :, k)*d(:, i, k); % s
        Dx_K_Sxh_K_Dx = Dx_K_Sxh_K_Dx + Dx(:, :, k, i)'*K(:, :, k)'*Sxh(:,:,k+1)*K(:, :, k)*Dx(:, :, k, i); % S
        Du_K_Sxh_K_d = Du_K_Sxh_K_d + Du(:, :, k, i)'*K(:, :, k)'*Sxh(:,:,k+1)*K(:, :, k)*d(:, i, k); % g
        Du_K_Sxh_K_Dx = Du_K_Sxh_K_Dx + Du(:, :, k, i)'*K(:, :, k)'*Sxh(:,:,k+1)*K(:, :, k)*Dx(:, :, k, i); % G
        Du_K_Sxh_K_Du = Du_K_Sxh_K_Du + Du(:, :, k, i)'*K(:, :, k)'*Sxh(:,:,k+1)*K(:, :, k)*Du(:, :, k, i); % H
    end

    AKF = A(:, :, k) - K(:, :, k)*F(:, :, k);
    
    % Calculate cost terms
    g  = r(:,k) + B(:,:,k)'*(sx(:,k+1) + sxh(:,k+1)) + Cu_Sx_c + Du_K_Sxh_K_d; % EQN (24) in Todorov 2007
    
    Gx = P(:,:,k)'  + B(:,:,k)'*(Sx(:,:,k+1) + Sxxh(:,:,k+1))*A(:,:,k) + B(:,:,k)'*(Sxh(:,:,k+1) + Sxxh(:,:,k+1))*K(:, :, k)*F(:, :, k) + Cu_Sx_Cx + Du_K_Sxh_K_Dx; % EQN (25) in Todorov 2007
    Gxh = B(:,:,k)'*(Sxh(:,:,k+1) + Sxxh(:,:,k+1))*AKF; % EQN (26) in Todorov 2007
    if ~isempty(fxu)

        fxuVx = vectens(sx(:,k+1) + sxh(:,k+1), fxu(:,:,:,k)); 
        Gx   = Gx + fxuVx;
    end
    

    H = R(:,:,k) + B(:,:,k)'*(Sx(:,:,k+1) + Sxh(:,:,k+1) + 2*Sxxh(:,:,k+1))*B(:,:,k) + Cu_Sx_Cu + Du_K_Sxh_K_Du; % EQN (23) in Todorov 2007
    if ~isempty(fuu)


        fuuVx = vectens(sx(:,k+1) + sxh(:,k+1), fuu(:,:,:,k)); 
        H   = H + fuuVx;
    end
    
    % lamdba regularization of G or H for calculation of gains 
    % (note: non-regularized terms are used in the updating of cost terms)

    S_reg = Sx(:,:,k+1) + Sxh(:,:,k+1) + 2*Sxxh(:,:,k+1) + lambda*eye(n)*(regType == 2); % option 2 for lambda reg, Tassa 2012
    
    G_reg = P(:,:,k)'  + B(:,:,k)'*S_reg*A(:,:,k) + Cu_Sx_Cx + Du_K_Sxh_K_Dx; % EQN (48) in Todorov 2007
    if ~isempty(fxu)
        G_reg = G_reg + fxuVx;
    end
    H_w_S_reg = R(:,:,k) + B(:,:,k)'*S_reg*B(:,:,k) + Cu_Sx_Cu + Du_K_Sxh_K_Du; % 
    H_reg = H_w_S_reg + lambda*eye(m)*(regType == 1) + (lambda - min(eig(H_w_S_reg)))*eye(m)*(regType == 3); % option 1 (Tassa 2012) and 3 (EQN (48) in Todorov 2007) for lambda reg, but no R in H from EQN (48) in Todorov 2007...
    
    if regType == 4 % Todorov 2007 p 1445
        [V, D] = eig(H_w_S_reg);
        D(D<lambda) = lambda;
        H_reg = V*D*V';
    end
    
    if ~isempty(fuu)
        H_reg = H_reg + fuuVx;
    end
    
    if isempty(lims) || lims(1,1) > lims(1,2) % nargin < 13 || isempty(lims) || lims(1,1) > lims(1,2)
        % no control limits: Cholesky decomposition, check for non-PD
        [H_reg_sqrt, flag] = chol(H_reg);
        if flag ~= 0
            diverge  = k;
            return;
        end
        
        % find control law
        lL = -H_reg_sqrt\(H_reg_sqrt'\[g G_reg]);
        l_i = lL(:,1);
        L_i = lL(:,2:n+1);

        
    else        % solve Quadratic Program

        [H_reg_sqrt, flag] = chol(H_reg);
        if flag ~= 0
            diverge  = k;
            return;
        end

        lower = lims(:,1)-u(:,k);
        upper = lims(:,2)-u(:,k);
        
        [l_i,result,H_reg_sqrt,free] = boxQP(H_reg,g,lower,upper,l(:,min(k+1,N-1)));
        
        if result < 1
            diverge  = -1*k; % negative to distinguish from chol() failing
            warning(['BoxQP chol() failed, result code =', num2str(result)])
            return;
        end
        
        L_i    = zeros(m,n);
        if any(free)
            Lfree        = -H_reg_sqrt\(H_reg_sqrt'\G_reg(free,:)); % was just G
            L_i(free,:)   = Lfree;
        end

    end
    


    % update cost-to-go approximation
    s0_alpha = s0_alpha + [l_i'*g, 0.5*l_i'*H*l_i]; % vector s0 terms for alpha backtracking reduction comparison

    Sx(:,:,k)  = Q(:,:,k)   + A(:,:,k)'*Sx(:,:,k+1)*A(:,:,k)  + F(:,:,k)'*K(:, :, k)'*Sxh(:,:,k+1)*K(:, :, k)*F(:,:,k)  + 2*A(:,:,k)'*Sxxh(:,:,k+1)*K(:, :, k)*F(:,:,k)   + Cx_Sx_Cx + Dx_K_Sxh_K_Dx;  % EQN 17) in Todorov 2007
    Sxh(:, :, k) = AKF'*Sxh(:, :, k+1)*AKF + L_i'*H*L_i        + L_i'*Gxh   + Gxh'*L_i; % EQN (18) in Todorov 2007
    Sxxh(:, :, k) = F(:,:,k)'*K(:, :, k)'*Sxh(:,:,k+1)*AKF + A(:,:,k)'*Sxxh(:,:,k+1)*AKF + Gx'*L_i; % EQN (19) in Todorov 2007

    sx(:,k) = q(:,k)     + A(:,:,k)'*sx(:,k+1) + + F(:, :, k)'*K(:, :, k)'*sxh(:, k) + Gx'*l_i     + Cx_Sx_c + Dx_K_Sxh_K_d; % EQN (20) in Todorov 2007, typo in paper as "d" should be bold
    sxh(:, k) = AKF'*sxh(:, k+1) + L_i'*H*l_i        + L_i'*g   + Gxh'*l_i; % EQN (21) in Todorov 2007
    
    s0(k) = q0(k) + s0(k+1) + 0.5*l_i'*H*l_i + l_i'*g + 0.5*(c_Sx_c + d_K_Sxh_K_d); % EQN (22) in Todorov 2007, typo in paper as "s" on LHS should not be bold


    if ~isempty(fxx)
        Sx(:,:,k) = Sx(:,:,k) + vectens(sx(:,k+1) + sxh(:,k+1), fxx(:,:,:,k)); 
    end

    
    % save controls/gains
    l(:,k)      = l_i;
    L(:,:,k)    = L_i;
end

end