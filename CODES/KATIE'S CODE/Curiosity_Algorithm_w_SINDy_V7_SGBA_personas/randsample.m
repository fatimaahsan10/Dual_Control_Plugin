function y = randsample(population, k, replace, weights)
%RANDSAMPLE  Minimal shim of MATLAB Statistics Toolbox randsample.
%  Supports the call pattern used in this codebase:
%    y = randsample(1:n, k, true, weights)
%  population : vector to sample from
%  k          : number of samples to draw
%  replace    : must be true (only with-replacement is implemented)
%  weights    : probability weights (need not be normalised)

if nargin < 3
    replace = false;
end
if ~replace
    error('randsample shim: only replace=true is implemented');
end

if nargin < 4 || isempty(weights)
    idx = randi(numel(population), k, 1);
else
    w = weights(:) / sum(weights);
    cdf = cumsum(w);
    r = rand(k, 1);
    idx = zeros(k, 1);
    for i = 1:k
        idx(i) = find(cdf >= r(i), 1, 'first');
    end
end

pop = population(:);
y = pop(idx)';
if iscolumn(population)
    y = y';
end
end
