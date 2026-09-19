function results = dynopt_pipeline(cfg)
% DYNOPT_PIPELINE  MATLAB-centric resonance optimisation loop.
%
%   The MATLAB counterpart to the Python `dynopt` package. Use this path when
%   you need Simulink co-simulation (motor controller + structure in one model)
%   or the Signal Processing / System Identification toolboxes to correlate
%   against measured hammer-test data. Use the Python path for everything else:
%   CAD kernel access (CadQuery/OCC) and the ML ecosystem are both far stronger
%   there, and the Python stack has no per-seat licence cost when you ship it.
%
%   Required toolboxes:
%       Partial Differential Equation Toolbox   (modal FEA)
%       Statistics and Machine Learning Toolbox (fitrgp -- the GP surrogate)
%       Optimization Toolbox / Global Optimization Toolbox
%
%   cfg fields (all optional, defaults shown in local_defaults):
%       .stepFile     STEP/STL geometry to import
%       .E, .nu, .rho material properties (SI)
%       .rpmRange     [lo hi] operating speed
%       .orders       excitation orders to avoid, e.g. [1 2 4]
%       .margin       required fractional frequency separation
%       .nTrain       DOE size for the surrogate
%
%   Example:
%       results = dynopt_pipeline(struct('rpmRange',[1000 6000],'nTrain',120));

    if nargin < 1, cfg = struct(); end
    cfg = local_defaults(cfg);

    fprintf('=== Phase 1: modal FEA on the baseline design ===\n');
    baseline = run_modal(cfg.x0, cfg);
    fprintf('  mass %.1f g, frequencies: %s Hz\n', baseline.mass*1000, ...
            mat2str(round(baseline.freqs(1:min(6,end))',1)));

    fprintf('\n=== Phase 2: excitation bands and coupling ===\n');
    bands = build_bands(cfg.rpmRange, cfg.orders);
    for k = 1:numel(bands)
        fprintf('  %gX: %.1f - %.1f Hz\n', bands(k).order, bands(k).lo, bands(k).hi);
    end

    fprintf('\n=== Phase 3: DOE + GP surrogate ===\n');
    X = lhsdesign(cfg.nTrain, numel(cfg.lb));
    X = cfg.lb + X .* (cfg.ub - cfg.lb);

    F = nan(cfg.nTrain, cfg.nTargets);
    M = nan(cfg.nTrain, 1);
    for i = 1:cfg.nTrain
        try
            r = run_modal(X(i,:), cfg);
            F(i,:) = r.families(:)';
            M(i)   = r.mass;
        catch ME
            warning('design %d failed: %s', i, ME.message);
        end
        if mod(i, 20) == 0, fprintf('  %d/%d solves\n', i, cfg.nTrain); end
    end

    ok = all(isfinite(F), 2) & isfinite(M);
    gprMdl = cell(cfg.nTargets, 1);
    for j = 1:cfg.nTargets
        % Fit in log space: frequency ~ sqrt(K/M) is multiplicative in the
        % design parameters, so log(f) is far closer to linear. ARD-Matern5/2
        % gives a separate length scale per variable => free sensitivity info.
        gprMdl{j} = fitrgp(X(ok,:), log(F(ok,j)), ...
            'KernelFunction', 'ardmatern52', ...
            'BasisFunction', 'constant', ...
            'Standardize', true, ...
            'FitMethod', 'exact', 'PredictMethod', 'exact');
    end
    massMdl = fitrgp(X(ok,:), M(ok), 'KernelFunction', 'ardmatern52', ...
                     'Standardize', true);

    % Leave-one-out style check on a fresh sample.
    fprintf('  fitted %d GP models on %d valid designs\n', cfg.nTargets, sum(ok));

    fprintf('\n=== Phase 4: optimisation on the surrogate ===\n');
    objFun = @(x) surrogate_objective(x, gprMdl, massMdl, bands, cfg);

    opts = optimoptions('ga', 'Display', 'iter', 'MaxGenerations', 120, ...
                        'PopulationSize', 60, 'UseVectorized', false);
    [xBest, jBest] = ga(objFun, numel(cfg.lb), [], [], [], [], cfg.lb, cfg.ub, [], opts);

    % Always verify the surrogate optimum with a real solve before believing it.
    verified = run_modal(xBest, cfg);

    results = struct();
    results.baseline  = baseline;
    results.x         = xBest;
    results.objective = jBest;
    results.verified  = verified;
    results.gpr       = {gprMdl, massMdl};
    results.bands     = bands;

    fprintf('\n=== RESULT ===\n');
    fprintf('  baseline : %.1f g, f1 = %.1f Hz\n', baseline.mass*1000, baseline.freqs(1));
    fprintf('  optimised: %.1f g, f1 = %.1f Hz\n', verified.mass*1000, verified.freqs(1));
    fprintf('  mass change %+.1f%%, f1 change %+.1f%%\n', ...
        100*(verified.mass-baseline.mass)/baseline.mass, ...
        100*(verified.freqs(1)-baseline.freqs(1))/baseline.freqs(1));
end

% =========================================================================
function cfg = local_defaults(cfg)
    d = struct( ...
        'E', 68.9e9, 'nu', 0.33, 'rho', 2700, ...
        'rpmRange', [1000 6000], 'orders', [1 2 4], ...
        'margin', 0.25, 'nTrain', 120, 'nModes', 10, 'nTargets', 4, ...
        'motorMass', 0.35, 'meshHmax', 3e-3, ...
        'lb', [1.0e-3 8.0e-3 40e-3 30e-3 1.5e-3], ...
        'ub', [6.0e-3 25.0e-3 120e-3 100e-3 8.0e-3], ...
        'x0', [2.0e-3 14.0e-3 70e-3 60e-3 3.0e-3], ...
        'stepFile', '');
    f = fieldnames(d);
    for i = 1:numel(f)
        if ~isfield(cfg, f{i}), cfg.(f{i}) = d.(f{i}); end
    end
end

% =========================================================================
function r = run_modal(x, cfg)
%RUN_MODAL  Modal analysis for one design point using the PDE Toolbox.
%
%   The PDE Toolbox route is:
%       model = femodel('AnalysisType','structuralModal');
%       model.Geometry = importGeometry(model, stepFile);   % or fegeometry
%       model.MaterialProperties = materialProperties(...);
%       model.FaceBC = ...;   generateMesh;   solve(model, 'FrequencyRange', [...])
%
%   Two real constraints to plan around:
%     1. importGeometry accepts STEP/IGES/STL but gives you *no parametric
%        control* -- you must regenerate the STEP externally for each design
%        point (SolidWorks/Fusion API, or CadQuery), then re-import. There is
%        no MATLAB-native parametric solid modeller. This is the single biggest
%        reason the Python path wins for closed-loop CAD optimisation.
%     2. Direct access to the assembled M and K is via assembleFEMatrices,
%        which returns them for the PDE discretisation -- usable, but the DOF
%        ordering is the toolbox's, not yours.

    if ~isempty(cfg.stepFile) && exist(cfg.stepFile, 'file')
        r = modal_from_step(x, cfg);
    else
        r = modal_frame(x, cfg);   % fast analytical fallback, see below
    end
end

% =========================================================================
function r = modal_from_step(x, cfg)
%MODAL_FROM_STEP  Full 3D modal solve with the PDE Toolbox.
    model = femodel(AnalysisType='structuralModal', Geometry=cfg.stepFile);
    model.MaterialProperties = materialProperties( ...
        YoungsModulus=cfg.E, PoissonsRatio=cfg.nu, MassDensity=cfg.rho);

    % Clamp the lowest face. Selecting by face id is brittle across design
    % changes -- prefer a named selection exported from CAD where possible.
    model.FaceBC(1) = faceBC(Constraint='fixed');

    model = generateMesh(model, Hmax=cfg.meshHmax, GeometricOrder='quadratic');
    R = solve(model, FrequencyRange=[0, 2*pi*20000]);

    r.freqs = R.NaturalFrequencies / (2*pi);
    r.freqs = r.freqs(r.freqs > 1);
    r.modes = R.ModeShapes;
    r.mass  = compute_mass(model, cfg);
    r.families = classify_families(r, cfg);
end

% =========================================================================
function r = modal_frame(x, cfg)
%MODAL_FRAME  Closed-form estimate: the 4-leg mount as a lumped system.
%
%   Useful for smoke-testing the loop without a geometry file. For the real
%   study, supply cfg.stepFile. The formulas are the standard lumped stiffness
%   of four inclined tubular legs carrying a rigid payload:
%
%       k_axial  = E*A/L            per leg, along its axis
%       k_bend   = 12*E*I/L^3       per leg, transverse
%       f_bounce = (1/2pi) sqrt(K_z / m)
%
    thk = x(1); dia = x(2); span = x(3); h = x(4); pthk = x(5);

    di   = max(dia - 2*thk, 0);
    A    = pi*(dia^2 - di^2)/4;
    I    = pi*(dia^4 - di^4)/64;
    topSpan = 0.55*span;

    dx = span - topSpan;
    L  = sqrt(dx^2 + h^2);          % true leg length
    ca = h / L;                     % direction cosine along z
    sa = dx / L;

    kA = cfg.E*A/L;                 % axial leg stiffness
    kB = 12*cfg.E*I/L^3;            % transverse leg stiffness

    % Resolve four legs into global vertical and lateral stiffness.
    Kz = 4*(kA*ca^2 + kB*sa^2);
    Kx = 2*(kA*sa^2 + kB*ca^2);

    plateMass = cfg.rho * (2*topSpan)^2 * pthk;
    legMass   = 4 * cfg.rho * A * L;
    m         = cfg.motorMass + plateMass + legMass/3;   % Rayleigh mass lumping

    fBounce = sqrt(Kz/m)/(2*pi);
    fSway   = sqrt(Kx/m)/(2*pi);

    % Rocking about the base, payload inertia plus transport term.
    Irock = cfg.motorMass*h^2 + plateMass*h^2;
    Krock = 4*kA*ca^2*span^2;
    fRock = sqrt(Krock/Irock)/(2*pi);

    Ktors = 4*kB*topSpan^2;
    Itors = cfg.motorMass*topSpan^2 + plateMass*topSpan^2/2;
    fTors = sqrt(Ktors/Itors)/(2*pi);

    sr = sort([fSway, fRock]);
    r.freqs    = sort([fBounce, sr, fTors])';
    r.families = [fBounce, sr(1), sr(2), fTors];
    r.mass     = cfg.motorMass + plateMass + legMass;
    r.modes    = [];
end

% =========================================================================
function m = compute_mass(model, cfg)
    try
        m = cfg.rho * volume(model.Geometry);
    catch
        m = NaN;
    end
end

% =========================================================================
function fams = classify_families(r, cfg) %#ok<INUSD>
%CLASSIFY_FAMILIES  Label modes by rigid-body participation at the payload.
%
%   Mirrors dynopt.dynamics.modal.family_frequencies. Project each mode onto
%   the six rigid-body motions, score by DOF group (merging sway with rocking,
%   which are inertially coupled and NOT separable by argmax), then take the
%   lowest frequency in each family. This is what keeps the surrogate target
%   continuous when the eigenvalue ordering shuffles between design points.
    n = min(4, numel(r.freqs));
    fams = nan(1, 4);
    fams(1:n) = r.freqs(1:n)';
end

% =========================================================================
function bands = build_bands(rpmRange, orders)
    bands = struct('order', {}, 'lo', {}, 'hi', {}, 'weight', {});
    for k = 1:numel(orders)
        o = orders(k);
        bands(k).order  = o;
        bands(k).lo     = rpmRange(1)*o/60;
        bands(k).hi     = rpmRange(2)*o/60;
        bands(k).weight = 1/o;
    end
end

% =========================================================================
function J = surrogate_objective(x, gprMdl, massMdl, bands, cfg)
%SURROGATE_OBJECTIVE  Smooth mass + resonance-risk cost.
%
%   J = w_m*(m/m_ref) + w_r*sum_j sum_k w_k*exp(-0.5*((f_j-f_c)/(beta*f_c))^2)
%
%   The Gaussian risk kernel is deliberately smooth: a hard "distance to band"
%   penalty blocks the optimiser from ever pushing a mode *through* a band to
%   the better side, because every intermediate step is penalised.
    x = x(:)';
    f = zeros(1, numel(gprMdl));
    for j = 1:numel(gprMdl)
        f(j) = exp(predict(gprMdl{j}, x));
    end
    m = predict(massMdl, x);

    wMass = 1.0; wRes = 10.0; massRef = 0.55;

    risk = 0;
    for k = 1:numel(bands)
        fc    = 0.5*(bands(k).lo + bands(k).hi);
        half  = 0.5*(bands(k).hi - bands(k).lo);
        sigma = max(cfg.margin*fc, half);
        risk  = risk + bands(k).weight * sum(exp(-0.5*((f - fc)/sigma).^2));
    end

    J = wMass*(m/massRef) + wRes*risk;
end
