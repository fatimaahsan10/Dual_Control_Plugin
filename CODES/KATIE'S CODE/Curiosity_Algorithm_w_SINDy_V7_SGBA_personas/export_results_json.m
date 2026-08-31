%% export_results_json.m
%  Exports all available sgba_persona_*.mat results to a single JSON file
%  for use by the manuscript-building scripts (Node/Python side).

keys = {'p1_maya','sophia','p3_riley','jean_guy','p5_daniel','joan', ...
        'kenny','p8_marcus','p9_aaliyah','p10_pierre','p11_niran','p12_lucia'};

TH = who_therapies();

out = struct();
out.therapy_codes = TH.codes;
out.therapy_names = TH.names;
out.personas = {};

for i = 1:numel(keys)
    k = keys{i};
    fname = fullfile('results', sprintf('sgba_persona_%s.mat', k));
    if ~exist(fname, 'file')
        fprintf('%-14s NOT YET AVAILABLE\n', k);
        continue;
    end
    load(fname);  % loads variable 'result'

    p = struct();
    p.key         = k;
    p.name        = result.persona_name;
    p.MFG         = result.MFG;
    p.final_scim  = result.final_scim;
    p.final_bbs   = result.final_bbs;
    p.swat        = result.swat;
    p.nmae        = result.nmae;
    p.lambda_opt  = result.lambda_opt;
    p.n_ilqg_fail = result.n_ilqg_fail;
    p.runtime_sec = result.runtime_sec;
    p.u_all       = result.u_all;        % 13 x 12
    p.scim_traj   = result.xa_all(1, 1:12);  % weeks 0-11 (col 13 unwritten by design)
    p.bbs_traj    = result.xa_all(2, 1:12);
    p.mean_dose   = result.mean_dose;    % 1 x 13

    early = mean(mean(result.u_all(:, 1:4)));
    late  = mean(mean(result.u_all(:, 9:12)));
    p.early_mean_dose = early;
    p.late_mean_dose  = late;
    if early > 1e-6
        p.taper_ratio = late / early;
    else
        p.taper_ratio = -1;  % undefined, flag with sentinel
    end

    out.personas{end+1} = p;
    fprintf('%-14s MFG=%.3f NMAE=%.4f fails=%d/12 taper=%.2f\n', ...
            k, p.MFG, p.nmae, p.n_ilqg_fail, p.taper_ratio);
end

fid = fopen('/home/claude/manuscript_build/persona_results.json', 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
fprintf('\nExported %d/%d personas to persona_results.json\n', numel(out.personas), numel(keys));
