function idx = crossvalind(method, N, K)
%CROSSVALIND  Minimal shim of MATLAB Statistics Toolbox crossvalind.
%  Supports: idx = crossvalind('Kfold', N, K)
%  Returns a column vector of fold assignments 1..K for N observations.

if ~strcmpi(method, 'Kfold')
    error('crossvalind shim: only ''Kfold'' is implemented');
end

base = mod((0:N-1)', K) + 1;
perm = randperm(N);
idx = zeros(N, 1);
idx(perm) = base;
end
