function F = compute_fisher_information(Theta, noise_std)
%COMPUTE_FISHER_INFORMATION  Compute the Fisher information matrix for the
%  SINDy parameter estimates, used by the dual iLQG curiosity term.
%
%  For a linear regression model  dX = Theta * Xi + epsilon,
%  epsilon ~ N(0, sigma^2 * I), the Fisher information matrix is:
%
%    F = (1 / sigma^2) * Theta' * Theta
%
%  This is the information matrix with respect to the coefficient vector Xi.
%  Its inverse gives the Cramer-Rao lower bound on parameter variance:
%
%    Sigma_Xi >= F^{-1}
%
%  The dual iLQG controller uses F to compute the expected information gain
%  from a proposed action, driving curiosity toward states where F grows
%  most rapidly (i.e., where parameter uncertainty is highest).
%
%  INPUTS
%    Theta      [N x n_terms]  library matrix (normalised)
%    noise_std  scalar         assumed observation noise standard deviation
%
%  OUTPUT
%    F          [n_terms x n_terms]  Fisher information matrix

sigma2 = noise_std^2;

if sigma2 < eps
    sigma2 = 1e-6;
    warning('compute_fisher_information: noise_std near zero. Using 1e-6.');
end

% Normalise Theta columns before computing FIM so that the information
% matrix is expressed in normalised-coefficient space (consistent with
% how STLSQ and the dual iLQG curiosity term operate).
col_std = std(Theta, 0, 1);
col_std(col_std < 1e-10) = 1;
Theta_n = Theta ./ col_std;

% Standard FIM for Gaussian linear model (in normalised space)
F = (1 / sigma2) * (Theta_n' * Theta_n);

% Symmetrise to correct floating-point asymmetry
F = (F + F') / 2;

% Check positive definiteness; report condition number
eigvals = eig(F);
if any(eigvals < 0)
    warning('compute_fisher_information: F is not positive semi-definite. Adding regularisation.');
    F = F + 1e-8 * eye(size(F));
end

end
