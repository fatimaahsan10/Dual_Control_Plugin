function H_PD = makePD(H, episilon, method)
% from Todorov 2007, p 1445

    % Method 1: Levenberg-Marquardt-like method
    if method == 1
        lambda_min = min(eig(H));
        H_PD = H + (episilon - lambda_min)*eye(size(H));
    
    % Method 2: Replace negative eigenvalues
    elseif method == 2
        [V, D] = eig(H);
        D_new = D;
        D_new(D_new < episilon) = episilon;
        H_PD = V*D_new*V';
    
    % Otherwise, error
    else
        disp('Method value not recognized')
        H_PD = NaN(size(H));
    end

end