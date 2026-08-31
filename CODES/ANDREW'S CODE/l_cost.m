%% l_cost.m
% Author: Andrew Mathis, andrew.mathis@unb.ca

% Based on Li and Todorov (2007): Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system


function c = l_cost(xa, u, Tracking_Trajectory, u_lims, u_lim_method, constants)
% cost function, sum of 3 terms:
% lu: quadratic cost on controls
% lf: final cost
% lx: running cost on states

% identify final time steps
final_index = find(isnan(u(1,:)));
final = isnan(u(1,:));
u(:,final)  = 0;


%% Cost coefficients

cu  = 1e-2.*[1 1000 75 75 50];  

cx  = 10.*[0.0033, 0.0267];
px  = [.1 .1]';             % smoothness scales for running cost, [1 nx]

cf  = 1e6*[0, 0];    % final cost coefficients, [1 nx] (NOT nxa as cost terms must all be independent of p)
pf  = [.1 .1]';    % smoothness scales for final cost, [1 nx]

cxc = 1e4; % x constraint cost

if u_lim_method == 2 && ~isempty(u_lims) % squashing method
    u = [u; (u_lims(:, 2) - u_lims(:, 1))/2.*tanh(u) + (u_lims(:, 2) + u_lims(:, 1))/2];
    cu  = [1e-6*ones(size(cu)), cu];         % control cost coefficients
    u(:,final)  = 0;
end


%% Calculate costs
% control cost

lu    = cu*u.^2; % for when no control limits

nx = length(cx);  % don't include parameters

lx = cx(1)*xa(6, :) + cx(2)*xa(8, :);

lxc = xa(6, :) > 15531/82999999;
lxc(lxc>0) = cxc;

% final cost
if any(final)
    llf      = cf*sabs([xa(6, final); xa(8, final)], pf); % use final index
    lf       = double(final);
    lf(final)= llf;
else
    lf    = 0;
end

% total cost
c     = lx + lu;
end

