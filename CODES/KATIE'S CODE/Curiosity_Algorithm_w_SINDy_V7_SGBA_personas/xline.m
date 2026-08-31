function h = xline(x, varargin)
%XLINE  Minimal shim of MATLAB's xline for Octave.
%  Draws a vertical line at x on the current axes. Accepts an optional
%  linespec string and label text/string, plus name-value pairs which are
%  passed through to line() where recognised.

linespec = '-';
extra = {};
i = 1;
if ~isempty(varargin) && ischar(varargin{1}) && ...
   any(strcmp(varargin{1}, {'-','--',':','-.','r--','k--','g--','b--','r-','k-'}))
    linespec = varargin{1};
    i = 2;
end
% Skip an optional label string (2nd positional arg in MATLAB's xline)
if i <= numel(varargin) && ischar(varargin{i}) && ...
   (i == numel(varargin) || ~ischar(varargin{i+1}) || mod(numel(varargin)-i,2)==1)
    % heuristic: treat as label only if remaining args look like name-value pairs
end

% Just pass any remaining name-value pairs straight through to line().
extra = varargin(i:end);
% Drop a leading label string if present (not all callers will have one,
% but the codebase only uses xline for cosmetic annotation lines).
if ~isempty(extra) && ischar(extra{1}) && mod(numel(extra)-1, 2) == 0 && numel(extra) > 1
    extra = extra(2:end);
elseif numel(extra) == 1 && ischar(extra{1})
    extra = {};
end

yl = ylim;
hold on;
h = line([x x], yl, 'LineStyle', regexprep(linespec, '[a-zA-Z]', ''), ...
         'Color', colorFromSpec(linespec));
if ~isempty(extra)
    try
        set(h, extra{:});
    catch
        % ignore unsupported property pass-through
    end
end
end

function c = colorFromSpec(spec)
    if any(spec == 'r'); c = 'r';
    elseif any(spec == 'g'); c = 'g';
    elseif any(spec == 'b'); c = 'b';
    elseif any(spec == 'k'); c = 'k';
    else; c = [0.3 0.3 0.3];
    end
end
