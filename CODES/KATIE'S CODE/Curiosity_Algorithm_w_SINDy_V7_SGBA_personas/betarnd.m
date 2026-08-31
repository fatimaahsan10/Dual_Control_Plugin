function r = betarnd(a, b, varargin)
%BETARND  Minimal shim of MATLAB Statistics Toolbox betarnd using the
%  Gamma-ratio method: if X~Gamma(a,1), Y~Gamma(b,1), then X/(X+Y)~Beta(a,b).
%
%  Supports: betarnd(a, b, m, n) and betarnd(a, b, [m n])

if nargin == 3 && isvector(varargin{1}) && numel(varargin{1}) == 2
    sz = varargin{1};
elseif nargin >= 4
    sz = [varargin{1}, varargin{2}];
else
    sz = [1, 1];
end

x = randg(a, sz);
y = randg(b, sz);
r = x ./ (x + y);
end
