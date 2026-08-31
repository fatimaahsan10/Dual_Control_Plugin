function yyaxis(side)
%YYAXIS  Minimal shim for Octave, which lacks true dual y-axes.
%  Falls back to operating on the current (single) axes so that scripts
%  written for MATLAB's yyaxis do not error out. Plot scaling on the
%  "right" side will share the same axis as "left" -- cosmetic only,
%  does not affect any computed/reported numbers.

if nargin < 1
    side = 'left';
end
ax = gca; %#ok<NASGU>  -- ensure an axes exists; nothing else to do.
end
