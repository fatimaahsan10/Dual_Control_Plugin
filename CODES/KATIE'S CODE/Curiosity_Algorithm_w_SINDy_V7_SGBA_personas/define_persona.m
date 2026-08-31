function persona = define_persona(name)
%DEFINE_PERSONA  Return a persona struct with clinical and behavioural
%  characteristics that modulate SCI rehabilitation dynamics and therapy
%  cost weighting.
%
%  Usage:
%    persona = define_persona('high_motivator');
%    persona = define_persona('frail_elder');
%    persona = define_persona('young_athlete');
%    persona = define_persona('low_support');
%    persona = define_persona('pain_sensitive');
%    persona = define_persona('default');
%
%  SGBA+ INTERSECTIONAL PERSONAS (added June 2026, Specific Aim 1 persona
%  development panel -- see Persona_development.xlsx for the full 45-attribute
%  source profiles). Each of these represents ONE concrete named/coded
%  patient (not a population archetype), so ais_probs is concentrated on
%  that patient's actual AIS grade rather than spread across a clinical
%  population:
%    persona = define_persona('p1_maya');      % P1  - 24, C1-C4 AIS A
%    persona = define_persona('sophia');       % P2  - 36, C1-C4 AIS B
%    persona = define_persona('p3_riley');     % P3  - 17, C1-C4 AIS D
%    persona = define_persona('jean_guy');     % P4  - 58, C5-T1 AIS A
%    persona = define_persona('p5_daniel');    % P5  - 42, C5-T1 AIS C
%    persona = define_persona('joan');         % P6  - 69, C5-T1 AIS D
%    persona = define_persona('kenny');        % P7  - 27, T2-T12 AIS A
%    persona = define_persona('p8_marcus');    % P8  - 25, T2-T12 AIS B
%    persona = define_persona('p9_aaliyah');   % P9  - 32, T2-T12 AIS D
%    persona = define_persona('p10_pierre');   % P10 - 40, L1-S5 AIS A
%    persona = define_persona('p11_niran');    % P11 - 75, L1-S5 AIS C
%    persona = define_persona('p12_lucia');    % P12 - 65, L1-S5 AIS D
%
%  Mapping rules used to derive fields below from the source SGBA+ profile
%  (documented here so the choices are auditable / clinician-reviewable):
%    ais_probs        90%+ mass on the persona's stated AIS grade (it is one
%                      named patient, not a sampled population)
%    age_range         tight band (+/- ~4-5 yrs) around age at evaluation
%    dpi_range         tight band reflecting care_continuum (community vs.
%                      tertiary/inpatient -> shorter DPI window)
%    caregiver_alpha/  derived from caregiver_inhome + family_supports +
%      beta            social_supports (strong in-home -> high alpha/low
%                      beta; none/institutional-only -> low alpha/high beta)
%    motivation        from pre-injury activity level, social capital,
%                      employment engagement, and any mental-health
%                      comorbidity (e.g. depression lowers motivation)
%    fatigue_tol       from age, cardiometabolic/respiratory comorbidities,
%                      and post-injury activity level
%    pain_sensitivity  from pain-management modality/intensity and
%                      comorbidities associated with chronic pain
%    comorbidity       from count/severity of comorbidities + other injuries
%                      sustained (e.g. concurrent TBI)
%    therapy_efficacy  from AIS grade + neuro level (residual motor/sensory
%                      function -> which WHO therapies are biomechanically
%                      relevant) and comorbidity profile (e.g. cardiac
%                      conditions de-prioritise high-load NMES/fitness)
%    w_resource        from distance to specialised centre, income, and
%                      paramedical/provider access (remote, low-income, or
%                      access-limited personas get a higher resource-
%                      preservation weight, biasing toward durable,
%                      lower-visit-burden therapy mixes)
%    w_scim / w_bbs    from injury level (cervical -> relatively higher
%                      w_scim for ADL/independence) and any persona-specific
%                      safety priority (e.g. living alone -> higher w_bbs)
%  NOTE: this mapping is a clinician-reviewable starting point, not a
%  validated clinical algorithm; it is the explicit object of the proposed
%  clinician-feedback iteration described in the accompanying manuscript.
%
%  Persona fields
%  ──────────────
%  Demographic / clinical priors (used in generate_synthetic_dataset):
%    ais_probs       [1x4]  Probability weights for AIS A/B/C/D
%    age_range       [1x2]  [min max] raw age (years)
%    dpi_range       [1x2]  [min max] days-post-injury at admission
%    caregiver_alpha scalar Beta-dist alpha for caregiver support
%    caregiver_beta  scalar Beta-dist beta  for caregiver support
%
%  Behavioural modifiers (used in generate_synthetic_dataset dynamics):
%    motivation      [0,1]  Amplifies response to therapy dose
%    fatigue_tol     [0,1]  Scales tolerance to high-load therapies
%    pain_sensitivity[0,1]  Attenuates high-load therapy response when high
%    comorbidity     [0,1]  Reduces functional ceiling (swat penalty)
%
%  THERAPY EFFICACY (NEW - June 2026):
%    therapy_efficacy [1x13]  Per-therapy responsiveness multiplier. A value
%      of 1.0 = population-average response; >1 = persona responds especially
%      well to that WHO therapy; <1 = poor responder. Ordered U1..U13 per
%      who_therapies(). THIS is the primary mechanism that makes the
%      curiosity-driven optimiser recommend a DIFFERENT therapy mix for each
%      persona - it is baked into the (persona-specific) identified dynamics,
%      so the controller discovers each persona's best levers on its own.
%
%  Cost-weight overrides (used in l_cost / Outer_Control_Loop):
%    w_scim          scalar Weight on SCIM gap in running cost
%    w_bbs           scalar Weight on BBS  gap in running cost
%    w_resource      scalar UNIFORM small penalty applied to every therapy
%                           (resource-preservation term - "doing anything
%                            costs something"). Replaces the old separate
%                            w_intensity / w_frequency weights.
%    w_terminal_scim scalar Terminal SCIM penalty
%    w_terminal_bbs  scalar Terminal BBS  penalty
%
%  Label:
%    name            string Human-readable persona label
%    description     string One-line clinical description
%
%  Author : Katie Campbell, UNB ECE
%  Date   : June 2026

%  Therapy-efficacy index reference (who_therapies order):
%   U1 ROM | U2 contracture-positioning | U3 strengthening | U4 NMES/FES |
%   U5 antispastic | U6 stretching | U7 balance | U8 gait |
%   U9 assistive-products | U10 mobility/wheelchair | U11 functional-positioning |
%   U12 hand/arm | U13 fitness

switch lower(strtrim(name))

    % -- 1. High Motivator -------------------------------------------------
    %  Young to mid-age, strong internal drive, tolerates intensive therapy.
    %  Broad strong responder; peaks on active strengthening/gait/balance.
    case 'high_motivator'
        persona.ais_probs        = [0.10, 0.15, 0.35, 0.40];
        persona.age_range        = [20, 50];
        persona.dpi_range        = [10, 40];
        persona.caregiver_alpha  = 3;
        persona.caregiver_beta   = 2;
        persona.motivation       = 0.90;
        persona.fatigue_tol      = 0.80;
        persona.pain_sensitivity = 0.20;
        persona.comorbidity      = 0.10;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.1, 1.0, 1.3, 1.2, 0.9, 1.0, 1.2, 1.3, 0.9, 1.0, 1.0, 1.2, 1.3];
        persona.w_scim           = 2.5;
        persona.w_bbs            = 1.2;
        persona.w_resource       = 5e-3;   % very low burden aversion
        persona.w_terminal_scim  = 60.0;
        persona.w_terminal_bbs   = 30.0;
        persona.name             = 'High Motivator';
        persona.description      = 'Young/mid-age, high drive, tolerates intensive regimens';

    % -- 2. Frail Elder ----------------------------------------------------
    %  Older adult, AIS A/B dominant, fatigue-limited, high comorbidity.
    %  Poor tolerance of high-load strengthening/NMES; thrives on balance,
    %  positioning, antispastic care and assistive-product training.
    case 'frail_elder'
        persona.ais_probs        = [0.45, 0.30, 0.15, 0.10];
        persona.age_range        = [60, 80];
        persona.dpi_range        = [20, 60];
        persona.caregiver_alpha  = 2;
        persona.caregiver_beta   = 3;
        persona.motivation       = 0.45;
        persona.fatigue_tol      = 0.25;
        persona.pain_sensitivity = 0.70;
        persona.comorbidity      = 0.65;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.0, 1.2, 0.6, 0.6, 1.2, 1.1, 1.3, 0.8, 1.3, 1.1, 1.2, 0.9, 0.7];
        persona.w_scim           = 1.5;
        persona.w_bbs            = 2.0;   % balance clinically critical
        persona.w_resource       = 0.20;  % strong resource/burden aversion
        persona.w_terminal_scim  = 35.0;
        persona.w_terminal_bbs   = 45.0;
        persona.name             = 'Frail Elder';
        persona.description      = 'Older adult, AIS A/B, fatigue-limited, high comorbidity';

    % -- 3. Young Athlete --------------------------------------------------
    %  Adolescent/young adult, high pre-injury fitness, rapid responder.
    %  Excels at high-load active therapy; less need for compensatory aids.
    case 'young_athlete'
        persona.ais_probs        = [0.05, 0.10, 0.40, 0.45];
        persona.age_range        = [16, 35];
        persona.dpi_range        = [5, 25];
        persona.caregiver_alpha  = 2.5;
        persona.caregiver_beta   = 2;
        persona.motivation       = 0.95;
        persona.fatigue_tol      = 0.95;
        persona.pain_sensitivity = 0.10;
        persona.comorbidity      = 0.05;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.1, 0.9, 1.5, 1.4, 0.8, 1.0, 1.2, 1.5, 0.7, 0.8, 0.9, 1.2, 1.5];
        persona.w_scim           = 3.0;
        persona.w_bbs            = 1.5;
        persona.w_resource       = 1e-3;  % almost no burden aversion
        persona.w_terminal_scim  = 80.0;
        persona.w_terminal_bbs   = 40.0;
        persona.name             = 'Young Athlete';
        persona.description      = 'Adolescent/young adult, high fitness pre-injury, rapid responder';

    % -- 4. Low Social Support ---------------------------------------------
    %  Mixed AIS, minimal caregiver. Favours therapies that build durable
    %  independence in-clinic (assistive products, wheelchair/mobility,
    %  hand/arm) over ones needing home reinforcement (ROM, stretching).
    case 'low_support'
        persona.ais_probs        = [0.25, 0.25, 0.25, 0.25];
        persona.age_range        = [25, 65];
        persona.dpi_range        = [15, 55];
        persona.caregiver_alpha  = 1.2;   % skewed toward low support
        persona.caregiver_beta   = 4.0;
        persona.motivation       = 0.60;
        persona.fatigue_tol      = 0.55;
        persona.pain_sensitivity = 0.40;
        persona.comorbidity      = 0.30;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 1.0, 1.0, 1.3, 1.3, 1.0, 1.2, 1.0];
        persona.w_scim           = 2.0;
        persona.w_bbs            = 1.0;
        persona.w_resource       = 8e-2;
        persona.w_terminal_scim  = 50.0;
        persona.w_terminal_bbs   = 25.0;
        persona.name             = 'Low Social Support';
        persona.description      = 'Mixed AIS, minimal caregiver, limited home reinforcement';

    % -- 5. Pain Sensitive -------------------------------------------------
    %  Chronic pain limits high-load escalation. Favours gentle therapies:
    %  positioning, antispastic care, stretching, functional positioning.
    case 'pain_sensitive'
        persona.ais_probs        = [0.20, 0.35, 0.35, 0.10];
        persona.age_range        = [30, 70];
        persona.dpi_range        = [20, 60];
        persona.caregiver_alpha  = 2;
        persona.caregiver_beta   = 2;
        persona.motivation       = 0.55;
        persona.fatigue_tol      = 0.40;
        persona.pain_sensitivity = 0.85;
        persona.comorbidity      = 0.50;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.0, 1.2, 0.6, 0.6, 1.3, 1.3, 1.0, 0.7, 1.1, 1.0, 1.2, 1.0, 0.7];
        persona.w_scim           = 1.8;
        persona.w_bbs            = 1.0;
        persona.w_resource       = 0.15;  % high burden from pain
        persona.w_terminal_scim  = 40.0;
        persona.w_terminal_bbs   = 20.0;
        persona.name             = 'Pain Sensitive';
        persona.description      = 'Chronic pain limits intensity escalation, AIS B/C';

    % -- 1. Maya (P1) -------------------------------------------------
    %  24, C1-C4 AIS A, urban, strong parental caregiver support, university student.
    case 'p1_maya'
        persona.ais_probs        = [0.94, 0.02, 0.02, 0.02];  % concentrated on AIS A (this is one named patient, not a population prior)
        persona.age_range        = [20, 28];
        persona.dpi_range        = [15, 45];
        persona.caregiver_alpha  = 4;
        persona.caregiver_beta   = 1.5;
        persona.motivation       = 0.65;
        persona.fatigue_tol      = 0.45;
        persona.pain_sensitivity = 0.45;
        persona.comorbidity      = 0.40;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.10, 1.30, 0.60, 1.20, 1.10, 1.10, 0.60, 0.50, 1.30, 1.20, 1.20, 1.30, 0.60];
        persona.w_scim           = 2.6;
        persona.w_bbs            = 0.8;
        persona.w_resource       = 0.04;
        persona.w_terminal_scim  = 55;
        persona.w_terminal_bbs   = 15;
        persona.name             = 'Maya (P1)';
        persona.description      = '24, C1-C4 AIS A, urban, strong parental caregiver support, university student';

    % -- 2. Sophia -------------------------------------------------
    %  36, C1-C4 AIS B, rural NB, assisted living, depression/anxiety, 110km from centre.
    case 'sophia'
        persona.ais_probs        = [0.02, 0.94, 0.02, 0.02];  % concentrated on AIS B (this is one named patient, not a population prior)
        persona.age_range        = [32, 40];
        persona.dpi_range        = [20, 60];
        persona.caregiver_alpha  = 1.3;
        persona.caregiver_beta   = 4.0;
        persona.motivation       = 0.40;
        persona.fatigue_tol      = 0.40;
        persona.pain_sensitivity = 0.55;
        persona.comorbidity      = 0.45;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 1.20, 0.50, 0.80, 1.20, 1.10, 1.00, 0.50, 1.30, 1.20, 1.20, 0.90, 0.50];
        persona.w_scim           = 2.0;
        persona.w_bbs            = 1.0;
        persona.w_resource       = 0.18;
        persona.w_terminal_scim  = 35;
        persona.w_terminal_bbs   = 20;
        persona.name             = 'Sophia';
        persona.description      = '36, C1-C4 AIS B, rural NB, assisted living, depression/anxiety, 110km from centre';

    % -- 3. Riley (P3) -------------------------------------------------
    %  17, C1-C4 AIS D, high pre-injury activity, lives with father, no comorbidities.
    case 'p3_riley'
        persona.ais_probs        = [0.02, 0.02, 0.02, 0.94];  % concentrated on AIS D (this is one named patient, not a population prior)
        persona.age_range        = [15, 20];
        persona.dpi_range        = [5, 25];
        persona.caregiver_alpha  = 3.5;
        persona.caregiver_beta   = 1.5;
        persona.motivation       = 0.85;
        persona.fatigue_tol      = 0.85;
        persona.pain_sensitivity = 0.20;
        persona.comorbidity      = 0.05;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 0.80, 1.30, 1.20, 0.70, 0.90, 1.30, 1.30, 0.70, 0.80, 0.80, 1.10, 1.20];
        persona.w_scim           = 2.8;
        persona.w_bbs            = 1.6;
        persona.w_resource       = 0.015;
        persona.w_terminal_scim  = 65;
        persona.w_terminal_bbs   = 35;
        persona.name             = 'Riley (P3)';
        persona.description      = '17, C1-C4 AIS D, high pre-injury activity, lives with father, no comorbidities';

    % -- 4. Jean Guy -------------------------------------------------
    %  58, C5-T1 AIS A, rural Acadian NB, diabetes/hypertension, 300km from centre, seasonal fisherman.
    case 'jean_guy'
        persona.ais_probs        = [0.94, 0.02, 0.02, 0.02];  % concentrated on AIS A (this is one named patient, not a population prior)
        persona.age_range        = [54, 63];
        persona.dpi_range        = [20, 60];
        persona.caregiver_alpha  = 2.2;
        persona.caregiver_beta   = 2.5;
        persona.motivation       = 0.55;
        persona.fatigue_tol      = 0.40;
        persona.pain_sensitivity = 0.55;
        persona.comorbidity      = 0.60;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 1.20, 0.50, 0.90, 1.20, 1.10, 0.70, 0.40, 1.30, 1.20, 1.20, 1.10, 0.50];
        persona.w_scim           = 2.2;
        persona.w_bbs            = 0.8;
        persona.w_resource       = 0.22;
        persona.w_terminal_scim  = 40;
        persona.w_terminal_bbs   = 15;
        persona.name             = 'Jean Guy';
        persona.description      = '58, C5-T1 AIS A, rural Acadian NB, diabetes/hypertension, 300km from centre, seasonal fisherman';

    % -- 5. Daniel (P5) -------------------------------------------------
    %  42, C5-T1 AIS C, urban, high income, computer programmer, high pre- and post-injury activity.
    case 'p5_daniel'
        persona.ais_probs        = [0.02, 0.02, 0.94, 0.02];  % concentrated on AIS C (this is one named patient, not a population prior)
        persona.age_range        = [38, 48];
        persona.dpi_range        = [10, 40];
        persona.caregiver_alpha  = 3.5;
        persona.caregiver_beta   = 1.5;
        persona.motivation       = 0.90;
        persona.fatigue_tol      = 0.80;
        persona.pain_sensitivity = 0.25;
        persona.comorbidity      = 0.10;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 0.80, 1.30, 1.20, 0.80, 0.90, 1.20, 1.20, 0.80, 0.90, 0.90, 1.10, 1.40];
        persona.w_scim           = 2.8;
        persona.w_bbs            = 1.4;
        persona.w_resource       = 0.005;
        persona.w_terminal_scim  = 70;
        persona.w_terminal_bbs   = 35;
        persona.name             = 'Daniel (P5)';
        persona.description      = '42, C5-T1 AIS C, urban, high income, computer programmer, high pre- and post-injury activity';

    % -- 6. Joan -------------------------------------------------
    %  69, C5-T1 AIS D, suburban, recently widowed, living alone, retired academic.
    case 'joan'
        persona.ais_probs        = [0.02, 0.02, 0.02, 0.94];  % concentrated on AIS D (this is one named patient, not a population prior)
        persona.age_range        = [65, 73];
        persona.dpi_range        = [10, 40];
        persona.caregiver_alpha  = 1.5;
        persona.caregiver_beta   = 4.0;
        persona.motivation       = 0.60;
        persona.fatigue_tol      = 0.55;
        persona.pain_sensitivity = 0.20;
        persona.comorbidity      = 0.20;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 1.10, 0.80, 0.90, 1.00, 1.00, 1.30, 1.00, 1.20, 1.10, 1.10, 1.00, 0.80];
        persona.w_scim           = 2.0;
        persona.w_bbs            = 2.0;
        persona.w_resource       = 0.1;
        persona.w_terminal_scim  = 50;
        persona.w_terminal_bbs   = 50;
        persona.name             = 'Joan';
        persona.description      = '69, C5-T1 AIS D, suburban, recently widowed, living alone, retired academic';

    % -- 7. Kenny -------------------------------------------------
    %  27, T2-T12 AIS A, rural, FASD/ADHD, low income, limited provider access, strong community ties.
    case 'kenny'
        persona.ais_probs        = [0.94, 0.02, 0.02, 0.02];  % concentrated on AIS A (this is one named patient, not a population prior)
        persona.age_range        = [23, 32];
        persona.dpi_range        = [15, 45];
        persona.caregiver_alpha  = 3.0;
        persona.caregiver_beta   = 2.0;
        persona.motivation       = 0.65;
        persona.fatigue_tol      = 0.65;
        persona.pain_sensitivity = 0.40;
        persona.comorbidity      = 0.35;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 0.90, 1.20, 1.00, 0.80, 0.90, 1.00, 0.60, 1.30, 1.40, 1.00, 1.20, 0.90];
        persona.w_scim           = 2.4;
        persona.w_bbs            = 0.8;
        persona.w_resource       = 0.2;
        persona.w_terminal_scim  = 50;
        persona.w_terminal_bbs   = 15;
        persona.name             = 'Kenny';
        persona.description      = '27, T2-T12 AIS A, rural, FASD/ADHD, low income, limited provider access, strong community ties';

    % -- 8. Marcus (P8) -------------------------------------------------
    %  25, T2-T12 AIS B, fused mid-back, construction foreman, no caregiver/family support, high substance use.
    case 'p8_marcus'
        persona.ais_probs        = [0.02, 0.94, 0.02, 0.02];  % concentrated on AIS B (this is one named patient, not a population prior)
        persona.age_range        = [22, 30];
        persona.dpi_range        = [10, 35];
        persona.caregiver_alpha  = 1.2;
        persona.caregiver_beta   = 4.5;
        persona.motivation       = 0.70;
        persona.fatigue_tol      = 0.75;
        persona.pain_sensitivity = 0.35;
        persona.comorbidity      = 0.30;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 1.00, 1.20, 1.20, 0.90, 1.00, 1.10, 1.00, 1.30, 1.30, 1.00, 1.10, 1.10];
        persona.w_scim           = 2.4;
        persona.w_bbs            = 1.2;
        persona.w_resource       = 0.12;
        persona.w_terminal_scim  = 55;
        persona.w_terminal_bbs   = 25;
        persona.name             = 'Marcus (P8)';
        persona.description      = '25, T2-T12 AIS B, fused mid-back, construction foreman, no caregiver/family support, high substance use';

    % -- 9. Aaliyah (P9) -------------------------------------------------
    %  32, T2-T12 AIS D, Brown-Sequard syndrome, diabetes, lives with roommate, active online community.
    case 'p9_aaliyah'
        persona.ais_probs        = [0.02, 0.02, 0.02, 0.94];  % concentrated on AIS D (this is one named patient, not a population prior)
        persona.age_range        = [28, 36];
        persona.dpi_range        = [10, 35];
        persona.caregiver_alpha  = 1.8;
        persona.caregiver_beta   = 3.0;
        persona.motivation       = 0.80;
        persona.fatigue_tol      = 0.55;
        persona.pain_sensitivity = 0.30;
        persona.comorbidity      = 0.30;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 1.00, 1.00, 0.90, 0.90, 1.00, 1.30, 1.20, 0.90, 1.00, 1.00, 1.10, 1.00];
        persona.w_scim           = 2.4;
        persona.w_bbs            = 1.6;
        persona.w_resource       = 0.1;
        persona.w_terminal_scim  = 55;
        persona.w_terminal_bbs   = 35;
        persona.name             = 'Aaliyah (P9)';
        persona.description      = '32, T2-T12 AIS D, Brown-Sequard syndrome, diabetes, lives with roommate, active online community';

    % -- 10. Pierre (P10) -------------------------------------------------
    %  40, L1-S5 AIS A, Quebec, electrician, heart condition, smoker, no primary care access.
    case 'p10_pierre'
        persona.ais_probs        = [0.94, 0.02, 0.02, 0.02];  % concentrated on AIS A (this is one named patient, not a population prior)
        persona.age_range        = [36, 45];
        persona.dpi_range        = [20, 60];
        persona.caregiver_alpha  = 3.0;
        persona.caregiver_beta   = 2.0;
        persona.motivation       = 0.50;
        persona.fatigue_tol      = 0.40;
        persona.pain_sensitivity = 0.40;
        persona.comorbidity      = 0.55;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 0.90, 1.10, 1.00, 0.70, 0.90, 0.90, 0.70, 1.20, 1.40, 1.00, 1.20, 0.60];
        persona.w_scim           = 2.4;
        persona.w_bbs            = 0.8;
        persona.w_resource       = 0.16;
        persona.w_terminal_scim  = 50;
        persona.w_terminal_bbs   = 15;
        persona.name             = 'Pierre (P10)';
        persona.description      = '40, L1-S5 AIS A, Quebec, electrician, heart condition, smoker, no primary care access';

    % -- 11. Niran (P11) -------------------------------------------------
    %  75, L1-S5 AIS C, dementia, lives with adult children, in-home caregiver, low baseline activity.
    case 'p11_niran'
        persona.ais_probs        = [0.02, 0.02, 0.94, 0.02];  % concentrated on AIS C (this is one named patient, not a population prior)
        persona.age_range        = [70, 80];
        persona.dpi_range        = [15, 50];
        persona.caregiver_alpha  = 3.5;
        persona.caregiver_beta   = 1.5;
        persona.motivation       = 0.40;
        persona.fatigue_tol      = 0.30;
        persona.pain_sensitivity = 0.40;
        persona.comorbidity      = 0.65;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.20, 1.20, 0.60, 0.70, 1.20, 1.10, 0.70, 0.50, 1.20, 1.00, 1.20, 0.80, 0.50];
        persona.w_scim           = 1.8;
        persona.w_bbs            = 1.2;
        persona.w_resource       = 0.14;
        persona.w_terminal_scim  = 35;
        persona.w_terminal_bbs   = 20;
        persona.name             = 'Niran (P11)';
        persona.description      = '75, L1-S5 AIS C, dementia, lives with adult children, in-home caregiver, low baseline activity';

    % -- 12. Lucia (P12) -------------------------------------------------
    %  65, L1-S5 AIS D, posterior cord syndrome, hypertension/obesity, widow, lives alone, retired librarian.
    case 'p12_lucia'
        persona.ais_probs        = [0.02, 0.02, 0.02, 0.94];  % concentrated on AIS D (this is one named patient, not a population prior)
        persona.age_range        = [60, 70];
        persona.dpi_range        = [15, 50];
        persona.caregiver_alpha  = 1.6;
        persona.caregiver_beta   = 3.5;
        persona.motivation       = 0.55;
        persona.fatigue_tol      = 0.40;
        persona.pain_sensitivity = 0.50;
        persona.comorbidity      = 0.50;
        %                          U1   U2   U3   U4   U5   U6   U7   U8   U9  U10  U11  U12  U13
        persona.therapy_efficacy = [1.00, 1.10, 0.70, 0.80, 1.10, 1.00, 1.40, 0.90, 1.10, 1.00, 1.10, 0.90, 0.60];
        persona.w_scim           = 1.8;
        persona.w_bbs            = 1.8;
        persona.w_resource       = 0.15;
        persona.w_terminal_scim  = 40;
        persona.w_terminal_bbs   = 40;
        persona.name             = 'Lucia (P12)';
        persona.description      = '65, L1-S5 AIS D, posterior cord syndrome, hypertension/obesity, widow, lives alone, retired librarian';

    % -- 18. Default (population average) ------------------------------------
    otherwise
        if ~strcmpi(name, 'default')
            warning('define_persona: unknown persona "%s". Using default.', name);
        end
        persona.ais_probs        = [0.30, 0.20, 0.25, 0.25];
        persona.age_range        = [18, 80];
        persona.dpi_range        = [10, 60];
        persona.caregiver_alpha  = 2;
        persona.caregiver_beta   = 2;
        persona.motivation       = 0.70;
        persona.fatigue_tol      = 0.60;
        persona.pain_sensitivity = 0.30;
        persona.comorbidity      = 0.25;
        persona.therapy_efficacy = ones(1, 13);   % uniform average response
        persona.w_scim           = 2.0;
        persona.w_bbs            = 1.0;
        persona.w_resource       = 1e-2;
        persona.w_terminal_scim  = 50.0;
        persona.w_terminal_bbs   = 25.0;
        persona.name             = 'Default';
        persona.description      = 'Original population-average parameters (no persona)';

end

%% -- Validate -------------------------------------------------------------

required = {'ais_probs','age_range','dpi_range','caregiver_alpha', ...
            'caregiver_beta','motivation','fatigue_tol','pain_sensitivity', ...
            'comorbidity','therapy_efficacy','w_scim','w_bbs','w_resource', ...
            'w_terminal_scim','w_terminal_bbs','name','description'};

for fi = 1:length(required)
    assert(isfield(persona, required{fi}), ...
        'define_persona: missing field "%s" in persona "%s"', required{fi}, name);
end

assert(abs(sum(persona.ais_probs) - 1) < 1e-6, ...
    'define_persona: ais_probs must sum to 1 for persona "%s"', name);

assert(numel(persona.therapy_efficacy) == 13, ...
    'define_persona: therapy_efficacy must have 13 entries (U1..U13) for "%s"', name);

end
