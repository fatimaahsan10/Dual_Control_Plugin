function h = sgtitle(txt, varargin)
%SGTITLE  Minimal shim of MATLAB's sgtitle for Octave.
%  Adds a super-title above all subplots using annotation-style text at
%  the top of the current figure. Cosmetic only.

try
    h = annotation('textbox', [0 0.94 1 0.06], 'String', txt, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold', 'FontSize', 12, varargin{:});
catch
    h = text(0.5, 0.98, txt, 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
end
