%% forward_pass.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

% Based on Li and Todorov (2007): Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system

function [A, B, c, Cx, Cu, d, Dx, Du, E, F, fxx, fxu, fuu, q0, q, Q, r, R, P]   = forward_pass(xa_bar, u_bar, constants, dynamics, measurement, cost, dt, full_DDP, nw, nv, ny, augment_states, dyn_noise_reg, lambda, Tracking_Trajectory, u_lims, u_lim_method)

% state and control indices
ixa = 1:size(xa_bar, 1);
iu = size(xa_bar, 1)+1:size(xa_bar, 1)+size(u_bar, 1);

N = size(xa_bar, 2);

%% dynamics derivatives
xu_f_dyn  = @(xau) dynamics(dt, xau(ixa,:), xau(iu,:), constants, zeros(nw, 1), false, augment_states, 0, dyn_noise_reg*lambda, u_lims, u_lim_method);
J_f       = finite_difference(xu_f_dyn, [xa_bar; u_bar]);
fx      = J_f(:,ixa,:);
fu      = J_f(:,iu,:);

A = fx; 
B = fu;

% dynamics second derivatives
if full_DDP
    xu_Jcst = @(xau) finite_difference(xu_f_dyn, xau);
    JJ = finite_difference(xu_Jcst, [xa_bar; u_bar]);
    JJ = reshape(JJ, [size(J_f,1) size(J_f,2) size(J_f,2) size(J_f,3)]); % is this right? See https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
    JJ = 0.5*(JJ + permute(JJ,[1 3 2 4])); %symmetrize
    fxx = JJ(:,ixa,ixa,:);
    fxu = JJ(:,ixa,iu,:);
    fuu = JJ(:,iu,iu,:);
else
    [fxx,fxu,fuu] = deal([]);
end

for i = 1:nw
    xu_F_dyn  = @(xau) dynamics(dt, xau(ixa,:), xau(iu,:), constants, zeros(nw, 1), i, augment_states, 0, dyn_noise_reg*lambda, u_lims, u_lim_method);
    c(:, i, :) = xu_F_dyn([xa_bar; u_bar]);
    J_F       = finite_difference(xu_F_dyn, [xa_bar; u_bar]);
    Fx(:, :, :, i)      = J_F(:,ixa,:);
    Fu(:, :, :, i)      = J_F(:,iu,:);
end
c = c*sqrt(dt);
Cx = Fx*sqrt(dt);
Cu = Fu*sqrt(dt);

%% measurement differentials
% Note: since calculating differentials here, yk input in Measurement given as zeroes
xu_g_dyn  = @(xau) measurement(dt, xau(ixa,:), xau(iu,:), constants, zeros(nv, size(xau, 2)), false, augment_states, 0);
J_g       = finite_difference(xu_g_dyn, [xa_bar; u_bar]);
gx      = J_g(:,ixa,:);
gu      = J_g(:,iu,:);

F = gx;
E = gu;

for j = 1:nv
    xu_G_dyn  = @(xau) measurement(dt, xau(ixa,:), xau(iu,:), constants, zeros(nv, size(xau, 2)), j, augment_states, 0);
    d(:, j, :) = xu_G_dyn([xa_bar; u_bar]);
    J_G       = finite_difference(xu_G_dyn, [xa_bar; u_bar]);
    Gx(:, :, :, j)      = J_G(:,ixa,:);
    Gu(:, :, :, j)      = J_G(:,iu,:);
end

d = d/sqrt(dt); % [ny, nv, N+1] after transposed
d    = reshape(d,  [ny nv N]); % if error on this line, check measurement length 
Dx = Gx/sqrt(dt);
Du = Gu/sqrt(dt);

%% cost first derivatives
xu_cost = @(xau) cost(xau(ixa,:),xau(iu,:), Tracking_Trajectory, u_lims, u_lim_method, constants);
J       = squeeze(finite_difference(xu_cost, [xa_bar; u_bar]));
lx      = J(ixa,:); % If this line produces an error, ensure than the control cost coeeficients in the cost function are the same size as the number of controls. 
lu      = J(iu,:);

% cost second derivatives
xu_Jcst = @(xau) squeeze(finite_difference(xu_cost, xau));
JJ      = finite_difference(xu_Jcst, [xa_bar; u_bar]);
JJ      = 0.5*(JJ + permute(JJ,[2 1 3])); %symmetrize
lxx     = JJ(ixa,ixa,:);
lxu     = JJ(ixa,iu,:);
luu     = JJ(iu,iu,:);

% Cost terms
q0 = cost(xa_bar, u_bar, Tracking_Trajectory, u_lims, u_lim_method, constants);
q = lx;
Q = lxx;
r = lu;
R = luu;
P = lxu;

q0 = dt*q0; 
q = dt*q;
Q = dt*Q;
r = dt*r;
R = dt*R;
P = dt*P;



end