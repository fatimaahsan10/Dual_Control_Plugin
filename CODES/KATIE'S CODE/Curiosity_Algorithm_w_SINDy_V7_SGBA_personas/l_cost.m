function c = l_cost(xa, u, Tracking_Trajectory, u_lims, u_lim_method, constants)
%L_COST  Running and terminal cost for SCI rehabilitation planning.
%
%   cost  =  state cost  +  resource cost  ( + terminal cost at horizon end )
%
%     state    : weighted squared gap of SCIM and BBS from the recovery
%                ceiling (swat). Drives both toward swat.
%     resource : ONE small uniform penalty on every therapy dose
%                (w_resource * sum_j u_j^2) - "doing anything costs something".
%     terminal : same gap, larger weight, applied only at the final step.
%
%   State   : xa(1)=SCIM xa(2)=BBS xa(3)=AIS xa(4)=Age xa(5)=DPI xa(6)=Caregiver
%   Control : u(1..13) = WHO therapy doses U1..U13 (who_therapies.m)
%
%   All cost weights (w_scim, w_bbs, w_resource, w_terminal_*) and the
%   recovery ceiling are persona-specific and read from the global
%   SINDY_MODEL; see define_persona.m.

% Final-step controls arrive as NaN - zero them so they add no resource cost.
final = isnan(u(1, :));
u(:, final) = 0;

% Map controls to physical [0,1] dose when tanh-squashing is active.
if u_lim_method == 2 && ~isempty(u_lims)
    u_phys = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u) + ...
             (u_lims(:,2) + u_lims(:,1))/2;
else
    u_phys = u;
end

% Recovery ceiling + persona cost weights.
[swat, w] = get_cost_params();

% Gap from ceiling (shared by running and terminal cost).
gap_scim = swat - xa(1, :);
gap_bbs  = swat - xa(2, :);

% Running cost: state gap + uniform resource penalty.
% The tiny 1e-6*sum(u.^2) term keeps the iLQG Hessian well-conditioned where
% tanh saturates (it is negligible against the real costs).
lx = w.scim * gap_scim.^2 + w.bbs * gap_bbs.^2;
lu = w.resource * sum(u_phys.^2, 1) + 1e-6 * sum(u.^2, 1);

% Terminal cost: applied at the final step only.
lf        = zeros(1, size(xa, 2));
lf(final) = w.term_scim * gap_scim(final).^2 + w.term_bbs * gap_bbs(final).^2;

c = lx + lu + lf;

end


function [swat, w] = get_cost_params()
%GET_COST_PARAMS  Recovery ceiling and persona-aware cost weights, read from
%  the global SINDY_MODEL. Falls back to population-average defaults.
global SINDY_MODEL;

if ~isempty(SINDY_MODEL) && isfield(SINDY_MODEL, 'swat_ceiling')
    swat = SINDY_MODEL.swat_ceiling;
else
    swat = 1.0;
end

if ~isempty(SINDY_MODEL) && isfield(SINDY_MODEL, 'persona')
    p = SINDY_MODEL.persona;
    w.scim      = p.w_scim;
    w.bbs       = p.w_bbs;
    w.resource  = p.w_resource;
    w.term_scim = p.w_terminal_scim;
    w.term_bbs  = p.w_terminal_bbs;
else
    w.scim      = 2.0;
    w.bbs       = 1.0;
    w.resource  = 1e-2;   % small uniform resource-preservation penalty
    w.term_scim = 50.0;
    w.term_bbs  = 25.0;
end

end
