function print_identified_equations(Xi, lib_labels, state_names, cfg)
%PRINT_IDENTIFIED_EQUATIONS  Display the SINDy-identified governing
%  equations in human-readable form, stripped of normalisation scaling.
%
%  This function implements the clinician interpretability requirement
%  from Section IV-E of the manuscript: the sparse differential equations
%  are printed with their physically meaningful variable names and
%  coefficient magnitudes, enabling physiotherapists to inspect, critique,
%  and override the inferred patient dynamics.
%
%  INPUTS
%    Xi           [n_terms x n_states]  sparse coefficient matrix (normalised)
%    lib_labels   {1 x n_terms}         library term labels (with scale tags)
%    state_names  {1 x n_states}        state variable names
%    cfg          configuration struct  (.verbose)

[n_terms, n_states] = size(Xi);

% Strip scale tags from labels and use Xi directly (already physical units)
raw_labels = cell(1, n_terms);
for k = 1:n_terms
    tok = regexp(lib_labels{k}, '^(.*?)\s*\[s=[\d.]+\]$', 'tokens');
    if ~isempty(tok)
        raw_labels{k} = strtrim(tok{1}{1});
    else
        raw_labels{k} = lib_labels{k};
    end
end

% Xi is already in physical units — use directly
Xi_phys = Xi;

fprintf('\n');
fprintf('══════════════════════════════════════════════════════════════════\n');
fprintf('  SINDy Identified Equations (physical coefficients)\n');
fprintf('══════════════════════════════════════════════════════════════════\n');

% Only print dynamic state equations (SCIM and BBS, indices 1 & 2)
% Static states (AIS, Age, DPI, Caregiver) have been zeroed and are skipped.
dynamic_states = 1:min(2, n_states);

for s = dynamic_states
    xi_s    = Xi_phys(:, s);
    nonzero = find(abs(xi_s) > 1e-8);

    fprintf('\n  d(%s)/dt = ', state_names{s});

    if isempty(nonzero)
        fprintf('0  (all coefficients zeroed)\n');
        continue;
    end

    terms_str = '';
    for k = 1:length(nonzero)
        idx   = nonzero(k);
        coeff = xi_s(idx);
        term  = raw_labels{idx};

        if k == 1
            terms_str = sprintf('%+.4f * %s', coeff, term);
        else
            terms_str = [terms_str, sprintf('  %+.4f * %s', coeff, term)]; %#ok<AGROW>
        end
    end
    fprintf('%s\n', terms_str);

    % Clinical interpretation hint
    fprintf('    [Active terms: %d / %d  |  ', length(nonzero), n_terms);
    print_clinical_interpretation(raw_labels(nonzero), xi_s(nonzero), state_names{s});
    fprintf(']\n');
end

fprintf('\n');
Xi_dyn = Xi(:, dynamic_states);
fprintf('  Non-zero coefficients (dynamic states): %d / %d  (sparsity: %.1f%%)\n', ...
        nnz(Xi_dyn), numel(Xi_dyn), 100*(1 - nnz(Xi_dyn)/numel(Xi_dyn)));
fprintf('══════════════════════════════════════════════════════════════════\n\n');

end

%% ── Local helper ─────────────────────────────────────────────────────────

function print_clinical_interpretation(labels, coeffs, state_name)
%  Print brief clinical interpretation of the dominant retained terms.

[~, sort_idx] = sort(abs(coeffs), 'descend');
top_n = min(2, length(labels));

parts = {};
for k = 1:top_n
    lbl = labels{sort_idx(k)};
    coeff = coeffs(sort_idx(k));
    direction = 'increases';
    if coeff < 0
        direction = 'decreases';
    end

    % Map label to clinical meaning.
    % Therapies a1..a13 -> WHO codes U1..U13 (regexp with word boundary so
    % that a1 does not partially match a10..a13).
    lbl_clean = regexprep(lbl, '\^2', ' squared');
    lbl_clean = regexprep(lbl_clean, 'a(\d+)', 'U$1');   % a7 -> U7, a13 -> U13
    lbl_clean = regexprep(lbl_clean, 'x1\>', 'SCIM');
    lbl_clean = regexprep(lbl_clean, 'x2\>', 'BBS');
    lbl_clean = regexprep(lbl_clean, 'x3\>', 'AIS');
    lbl_clean = regexprep(lbl_clean, 'x4\>', 'Age');
    lbl_clean = regexprep(lbl_clean, 'x5\>', 'DPI');
    lbl_clean = regexprep(lbl_clean, 'x6\>', 'Caregiver');

    parts{end+1} = sprintf('%s %s %s', lbl_clean, direction, state_name); %#ok<AGROW>
end

fprintf('%s', strjoin(parts, '; '));
end
