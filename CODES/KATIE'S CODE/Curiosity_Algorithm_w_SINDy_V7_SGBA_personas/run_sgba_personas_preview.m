%% =========================================================================
%  run_sgba_personas_preview.m
%  FAST PREVIEW: runs 3 contrasting SGBA+ personas with a 6-week horizon
%  (instead of 12) to validate the pipeline end-to-end before committing to
%  the full 12-persona x 12-week batch (run_sgba_personas.m). The per-week
%  iLQG solve dominates runtime (~25-30s/week in this environment) and scales
%  with persona count x session count, so this preview is roughly 1/8th the
%  cost of the full run: 3 personas x 6 weeks vs. 12 personas x 12 weeks.
%
%  Author : Katie Campbell, UNB ECE (preview scaffolding added June 2026)
% =========================================================================

clear; clc;
clear global SINDY_MODEL;
rng(7);

TH  = who_therapies();
n_u = TH.n;

% Three deliberately contrasting personas: a cervical/high-support/young
% patient, a thoracic/remote/low-resource patient, and a lumbar/elderly/
% comorbid patient -- chosen to stress-test that the pipeline differentiates
% plans across very different profiles before running the full set of 12.
persona_keys = {'p1_maya', 'jean_guy', 'p11_niran'};
n_personas = numel(persona_keys);

out_dir = 'results';
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

base_cfg = struct();
base_cfg.n_patients      = 40;
base_cfg.n_sessions      = 6;      % PREVIEW: 6-week horizon, not 12
base_cfg.noise_std       = 0.05;
base_cfg.poly_order      = 2;
base_cfg.include_trig    = false;
base_cfg.include_cross   = true;
base_cfg.lambda          = 0.05;
base_cfg.max_iter        = 100;
base_cfg.tol             = 1e-6;
base_cfg.nmae_threshold  = 1.0;
base_cfg.max_terms       = 22;
base_cfg.min_terms       = 3;
base_cfg.protected_terms = {'x3*a3', 'x6*a8'};
base_cfg.val_fraction    = 0.2;
base_cfg.n_lambda        = 8;
base_cfg.verbose         = false;
base_cfg.save_results    = false;

x0_true_base = [0.25; 0.20; 0.75; 0.45; 0.30; 0.70];

fprintf('=== PREVIEW: 3 personas, 6-week horizon, N=%d, n_lambda=%d ===\n', ...
        base_cfg.n_patients, base_cfg.n_lambda);

for pidx = 1:n_personas
    pkey  = persona_keys{pidx};
    fname = fullfile(out_dir, sprintf('preview_%s.mat', pkey));

    t0 = tic;
    persona = define_persona(pkey);
    fprintf('\n[%d/%d] %-12s  %s\n', pidx, n_personas, pkey, persona.name);

    cfg         = base_cfg;
    cfg.persona = persona;

    try
        dataset = generate_synthetic_dataset(cfg);

        n_val   = round(cfg.val_fraction * cfg.n_patients);
        n_train = cfg.n_patients - n_val;
        idx     = randperm(cfg.n_patients);
        train_idx = idx(1:n_train);
        val_idx   = idx(n_train+1:end);

        X_train = dataset.X(:, :, train_idx);
        A_train = dataset.A(:, :, train_idx);
        X_val   = dataset.X(:, :, val_idx);
        A_val   = dataset.A(:, :, val_idx);
        Y_val   = dataset.Y(:, :, val_idx);

        [Theta, dX, lib_labels, protected_idx] = build_sindy_library(X_train, A_train, cfg);
        lambda_opt = cross_validate_lambda(Theta, dX, cfg, protected_idx);
        cfg.lambda = lambda_opt;

        Xi = stlsq(Theta, dX, cfg.lambda, cfg.max_iter, cfg.tol, false, ...
                   cfg.max_terms, protected_idx, cfg.min_terms);
        [Xi, ~] = prune_sindy_bic(Xi, Theta, dX, lib_labels, ...
                                   dataset.state_names, false, protected_idx);
        static_states = 3:dataset.n_states;
        Xi(:, static_states) = 0;

        [nmae, ~, ~] = validate_sindy_model(Xi, X_val, A_val, Y_val, lib_labels, cfg);
        fprintf('  SINDy NMAE: %.4f (lambda*=%.4f)\n', nmae, lambda_opt);

        clear global SINDY_MODEL;
        global SINDY_MODEL;

        x0 = x0_true_base;
        swat = 0.4*x0(3) + 0.2*(1-x0(4)) + 0.2*x0(6) + 0.1*(1-x0(5));
        swat = swat * (1 - 0.5 * persona.comorbidity);
        swat = min(max(swat, 0.1), 1.0);

        np_max = 2;
        [term_idx, state_idx] = find(Xi(:, 1:2) ~= 0);
        n_uncertain = min(np_max, length(term_idx));
        xi_idx = [term_idx(1:n_uncertain), state_idx(1:n_uncertain)];
        np = n_uncertain;

        SINDY_MODEL.Xi           = Xi;
        SINDY_MODEL.lib_cfg      = cfg;
        SINDY_MODEL.swat_ceiling = swat;
        SINDY_MODEL.xi_idx       = xi_idx;
        SINDY_MODEL.F_dyn        = 1e-3;
        SINDY_MODEL.F_param      = 1e-4;
        SINDY_MODEL.persona      = persona;

        nx = 6; nu = n_u; ny = 2; nv = ny; nw = nx + np;
        N = cfg.n_sessions;                 % 6 for preview
        horizon_length = min(6, N);
        u_lims = repmat([0, 1], nu, 1);
        u_lim_method = 2;

        xi_true = zeros(np, 1);
        for k = 1:np
            xi_true(k) = Xi(xi_idx(k,1), xi_idx(k,2));
        end

        xa_true = [x0; xi_true];
        c_vec   = [swat; 1e-3; 1e-4];

        x_hat = x0;
        p_hat = xi_true;
        cov_xa = blkdiag(0.01*eye(nx), diag((0.8*abs(xi_true)+0.05).^2));

        u_all  = zeros(nu, N);
        xa_all = zeros(nx+np, N+1);
        xa_all(:,1) = xa_true;

        l_gain = 0.5 * ones(nu, horizon_length);
        L_gain = zeros(nu, nx+np, horizon_length);
        u_bar  = 0.5 * ones(nu, horizon_length);

        n_ilqg_fail = 0;

        for s = 1:N
            T_rem = min(horizon_length, N - s + 1);
            wk_t0 = tic;
            try
                [~, unew, l_gain_s, L_gain_s, ~, ~, ~, ~, ~, ~] = ...
                    iLQG_function(T_rem, 1, x_hat, l_gain(:,1:T_rem), ...
                                  L_gain(:,:,1:T_rem), u_bar(:,1:T_rem), ...
                                  1, 1, c_vec, p_hat, cov_xa, ...
                                  1, 3, u_lims, ny, nv, nw, 20, false, ...
                                  0, 0, [], u_lim_method);
                u_raw  = unew(:,1);
                u_phys = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u_raw) + ...
                         (u_lims(:,2) + u_lims(:,1))/2;
                u_all(:,s) = u_phys;
                l_gain(:,1:T_rem) = l_gain_s;
            catch
                n_ilqg_fail = n_ilqg_fail + 1;
                u_all(:,s) = 0.5 * ones(nu,1);
            end
            fprintf('    week %2d/%2d solved (%.1fs)\n', s, N, toc(wk_t0));

            if s < N
                sqrtR = 1e-2 * eye(nv);
                sqrtQ = blkdiag(1e-2*eye(nx), 1e-3*eye(np));
                [~, xa_sim] = simulate_system(1, xa_all(:,s), u_all(:,s), ...
                                 c_vec, sqrtR, sqrtQ, 1, 0, u_lims, u_lim_method);
                xa_all(:,s+1) = xa_sim(:,end);
                x_hat = xa_all(1:nx, s+1);
            end

            if T_rem > 1
                u_bar_raw  = [unew(:,2:end), unew(:,end)];
                u_bar_phys = (u_lims(:,2) - u_lims(:,1))/2 .* tanh(u_bar_raw) + ...
                             (u_lims(:,2) + u_lims(:,1))/2;
                u_bar = u_bar_phys;
                if size(u_bar,2) < horizon_length
                    u_bar = [u_bar, repmat(u_bar(:,end), 1, horizon_length-size(u_bar,2))];
                end
            end
        end

        final_scim = xa_all(1, N);
        final_bbs  = xa_all(2, N);
        init_score  = (x0(1) + x0(2)) / 2;
        final_score = (final_scim + final_bbs) / 2;
        ceiling_gap = max(swat - init_score, 0.01);
        MFG = (final_score - init_score) / ceiling_gap;
        mean_dose = mean(u_all, 2)';

        result = struct();
        result.persona_key  = pkey;
        result.persona_name = persona.name;
        result.persona      = persona;
        result.MFG          = MFG;
        result.final_scim   = final_scim;
        result.final_bbs    = final_bbs;
        result.swat         = swat;
        result.nmae         = nmae;
        result.lambda_opt   = lambda_opt;
        result.n_ilqg_fail  = n_ilqg_fail;
        result.mean_dose    = mean_dose;
        result.u_all        = u_all;
        result.xa_all       = xa_all;
        result.TH           = TH;
        result.runtime_sec  = toc(t0);

        save(fname, 'result');
        fprintf('  MFG=%.3f  SCIM=%.3f  BBS=%.3f  SWAT=%.3f  iLQG fails=%d/%d  (%.1fs total)\n', ...
                MFG, final_scim, final_bbs, swat, n_ilqg_fail, N, result.runtime_sec);
        fprintf('  saved -> %s\n', fname);

    catch err
        fprintf('  *** FAILED: %s ***\n', err.message);
    end
end

fprintf('\n=== Preview complete ===\n');
