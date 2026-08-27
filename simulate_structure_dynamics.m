%% STRUCTURE-DYNAMICS PREDICTIVE INFORMATION: IN SILICO VALIDATION
% Reproduces the synthetic-network analysis used to compare three model classes:
%
%   1) Nonlinear recurrent coupling
%          dx = [-x + M*tanh(x)] dt + sigma dB
%
%   2) Matched linear recurrent coupling
%          dx = [-x + Mbar*x] dt + sigma dB
%      where Mbar = M*Dbar and Dbar is the time-averaged nonlinear gain.
%      This retains recurrent structural coupling while removing instantaneous
%      state dependence from the local Jacobian.
%
%   3) Structural-drive control (no recurrent coupling)
%          dx_i = [-x_i + beta*S_i] dt + sigma dB_i
%      where structure changes the regional operating point but not local
%      state sensitivity.
%
% For each independently generated circular network, the script estimates
%
%          I(S ; X_future | X_present)
%
% at tau = 0.2 time units. S is a binned regional structural descriptor
% (weighted anatomical strength). Significance is assessed against dihedral
% spatial surrogates (rotations/reflections) of S, which preserve its circular
% spatial organization while disrupting the region-to-structure assignment.
%
% The normalized contribution
%
%          eta = I(S ; X_future | X_present) / H(X_future | X_present)
%
% is also reported as the fraction of residual uncertainty explained by S.
%
% Reproducibility:
%   - All random-number streams are explicitly seeded.
%   - The script is self-contained; helper functions are defined below.
%   - MATLAB's signrank() is used when the Statistics and Machine Learning
%     Toolbox is available; the exact sign test is always reported.
%
% Output:
%   structure_dynamics_simulation_results.mat
%
% Notes:
%   - The simulation itself does NOT estimate reactivity/perturbation growth.
%     Reactivity is a mechanistic implication discussed separately in the paper.
%   - Across-network summaries below are reported as mean +/- SEM. SD is also
%     saved in the output file for transparency.

clear;
close all;
clc;

%% ------------------------------------------------------------------------
% Parameters
% -------------------------------------------------------------------------

baseSeed = 2000;

N = 60;                    % number of regions on the circular network
nNetworks = 30;            % independently generated structural networks
nRealizations = 5;         % stochastic trajectories per network and model

dt = 0.01;                % Euler-Maruyama integration step
T = 300;                   % total simulated duration
burnIn = 100;              % discarded transient period

dtInfo = 0.20;             % sampling interval used for information analysis
infoStride = round(dtInfo / dt);

tau = 0.20;                % prediction horizon
lag = round(tau / dtInfo);

infoThin = 5;               % additional thinning of information samples
sigmaNoise = 0.20;          % additive Gaussian noise amplitude

targetEffectiveRadius = 2.5;
fractionNegative = 0.30;    % fraction of inhibitory/signed source efficacies
negativeStrength = -3;
lengthScale = 0.7;          % spatial decay scale of anatomical weights

nBinsX = 5;                 % quantile bins for present/future state
nBinsS = 3;                 % quantile bins for structural descriptor

betaControl = 0.6;          % structural-drive control coefficient

% Basic consistency checks.
assert(abs(infoStride * dt - dtInfo) < 1e-12, ...
    'dtInfo must be an integer multiple of dt.');
assert(abs(lag * dtInfo - tau) < 1e-12, ...
    'tau must be an integer multiple of dtInfo.');
assert(burnIn < T, 'burnIn must be shorter than T.');

%% ------------------------------------------------------------------------
% Time bookkeeping
% -------------------------------------------------------------------------

Nt = round(T / dt) + 1;
burnStep = round(burnIn / dt) + 1;
infoSteps = burnStep:infoStride:Nt;
nInfo = numel(infoSteps);

%% ------------------------------------------------------------------------
% Preallocate network-level results
% -------------------------------------------------------------------------

I_nonlin = zeros(nNetworks, 1);
eta_nonlin = zeros(nNetworks, 1);
p_nonlin = zeros(nNetworks, 1);

I_linear = zeros(nNetworks, 1);
eta_linear = zeros(nNetworks, 1);
p_linear = zeros(nNetworks, 1);

I_c0 = zeros(nNetworks, 1);
eta_c0 = zeros(nNetworks, 1);
p_c0 = zeros(nNetworks, 1);

delta_nonlin_linear = zeros(nNetworks, 1);
delta_linear_c0 = zeros(nNetworks, 1);

meanGainNetwork = zeros(nNetworks, 1);
g_all = zeros(nNetworks, 1);

%% ========================================================================
% Network loop
% =========================================================================

for net = 1:nNetworks

    fprintf('\n====================================================\n');
    fprintf('NETWORK %d / %d\n', net, nNetworks);
    fprintf('====================================================\n');

    % One reproducible structural network per network index.
    rng(baseSeed + 10000 * net);

    %% --------------------------------------------------------------------
    % Construct a weighted circular anatomical network W
    % ---------------------------------------------------------------------

    theta = linspace(0, 2*pi, N + 1);
    theta(end) = [];

    dist = zeros(N);
    for i = 1:N
        for j = 1:N
            d = abs(theta(i) - theta(j));
            dist(i,j) = min(d, 2*pi - d);
        end
    end

    % Distance-dependent anatomical coupling.
    W = exp(-dist / lengthScale);

    % Heterogeneous regional strength factors.
    nodeStrengthFactor = exp(0.35 * randn(N,1));
    W = W .* (nodeStrengthFactor * nodeStrengthFactor');

    % Symmetric pairwise heterogeneity.
    randomFactor = 0.7 + 0.6 * rand(N);
    randomFactor = 0.5 * (randomFactor + randomFactor');
    W = W .* randomFactor;

    % Exclude self-connections and normalize the anatomical matrix.
    W(1:N+1:end) = 0;
    rhoW = max(abs(eig(W)));
    W = W / rhoW;

    assert(all(W(:) >= -1e-12), 'Anatomical W should be non-negative.');
    assert(norm(W - W', 'fro') < 1e-10, 'Anatomical W should be symmetric.');

    %% --------------------------------------------------------------------
    % Signed physiological source efficacy
    % ---------------------------------------------------------------------
    % W is anatomical and non-negative. Signs are introduced separately at
    % the source level. Multiplication by sigmaSource' therefore scales the
    % columns of W, consistent with M(i,j) multiplying activity from source j.

    sigmaSource = ones(N,1);
    nNegative = round(fractionNegative * N);
    negativeIdx = randperm(N, nNegative);
    sigmaSource(negativeIdx) = negativeStrength;

    Wsigma = W .* sigmaSource';

    %% --------------------------------------------------------------------
    % Choose global nonlinear coupling g
    % ---------------------------------------------------------------------

    rhoBase = max(abs(eig(Wsigma)));
    g = targetEffectiveRadius / rhoBase;
    g_all(net) = g;

    M = g * Wsigma;

    fprintf('g = %.4f\n', g);
    fprintf('rho(M) = %.4f\n', max(abs(eig(M))));

    %% --------------------------------------------------------------------
    % Structural descriptor S and circular spatial null
    % ---------------------------------------------------------------------
    % Because W is symmetric, row strength and column strength are identical.
    % S is discretized into quantile bins for the CMI estimator.

    S = sum(W, 2);
    S_z = (S - mean(S)) / std(S);
    Sbin = quantile_bin(S, nBinsS);

    % Dihedral null: all non-identity rotations plus rotations of the
    % reflected structural map. Duplicate maps and the original map are
    % removed before testing.
    Ssur = zeros(N, 0);

    for k = 1:N-1
        Ssur(:,end+1) = circshift(Sbin, k); %#ok<SAGROW>
    end

    Sref = flipud(Sbin);
    for k = 0:N-1
        Ssur(:,end+1) = circshift(Sref, k); %#ok<SAGROW>
    end

    Ssur = unique(Ssur', 'rows', 'stable')';
    keep = true(1, size(Ssur,2));

    for s = 1:size(Ssur,2)
        if isequal(Ssur(:,s), Sbin)
            keep(s) = false;
        end
    end

    Ssur = Ssur(:, keep);

    assert(~isempty(Ssur), 'No non-identity spatial surrogates were generated.');

    %% ====================================================================
    % 1) Nonlinear recurrently coupled model
    % =====================================================================

    x_nonlin = zeros(N, nInfo, nRealizations);
    gainAccum = zeros(N,1);
    gainCount = 0;

    for r = 1:nRealizations

        % Separate reproducible stream for each network/realization.
        rng(baseSeed + 100000 * net + r);

        x = 0.5 * randn(N,1);
        infoCounter = 0;

        for step = 1:Nt

            if step >= burnStep

                % tanh'(x) = 1 - tanh(x)^2. The time-averaged gain is later
                % used to construct the matched linear control.
                gainAccum = gainAccum + (1 - tanh(x).^2);
                gainCount = gainCount + 1;

                if mod(step - burnStep, infoStride) == 0
                    infoCounter = infoCounter + 1;
                    x_nonlin(:,infoCounter,r) = x;
                end
            end

            if step == Nt
                break;
            end

            drift = -x + M * tanh(x);
            x = x + drift * dt ...
                + sigmaNoise * sqrt(dt) * randn(N,1);
        end

        assert(infoCounter == nInfo, 'Unexpected nonlinear sample count.');
    end

    %% --------------------------------------------------------------------
    % Time-averaged nonlinear gain and matched linear operator
    % ---------------------------------------------------------------------

    meanGainVector = gainAccum / gainCount;
    meanGainNetwork(net) = mean(meanGainVector);

    % Mbar = M*Dbar, with Dbar = diag(meanGainVector). Column scaling is
    % written explicitly to avoid forming a dense diagonal matrix.
    Mbar = M .* meanGainVector';
    Alinear = -eye(N) + Mbar;
    alphaLinear = max(real(eig(Alinear)));

    fprintf('Mean nonlinear gain = %.4f\n', meanGainNetwork(net));
    fprintf('Matched linear spectral abscissa = %.4f\n', alphaLinear);

    %% --------------------------------------------------------------------
    % Information analysis: nonlinear model
    % ---------------------------------------------------------------------

    [I_nonlin(net), eta_nonlin(net), p_nonlin(net)] = ...
        analyze_standardized_cmi( ...
            x_nonlin, Sbin, Ssur, lag, infoThin, nBinsX, nBinsS);

    %% ====================================================================
    % 2) Matched linear recurrently coupled control
    % =====================================================================

    x_linear = zeros(N, nInfo, nRealizations);

    for r = 1:nRealizations

        rng(baseSeed + 400000 + 100000 * net + r);

        x = 0.5 * randn(N,1);
        infoCounter = 0;

        for step = 1:Nt

            if step >= burnStep && mod(step - burnStep, infoStride) == 0
                infoCounter = infoCounter + 1;
                x_linear(:,infoCounter,r) = x;
            end

            if step == Nt
                break;
            end

            drift = -x + Mbar * x;
            x = x + drift * dt ...
                + sigmaNoise * sqrt(dt) * randn(N,1);
        end

        assert(infoCounter == nInfo, 'Unexpected linear sample count.');
    end

    [I_linear(net), eta_linear(net), p_linear(net)] = ...
        analyze_standardized_cmi( ...
            x_linear, Sbin, Ssur, lag, infoThin, nBinsX, nBinsS);

    %% ====================================================================
    % 3) Structural-drive control (no recurrent structural coupling)
    % =====================================================================
    % Structure affects the regional operating point through betaControl*S_z,
    % but the Jacobian of the dynamics is simply -I.

    x_c0 = zeros(N, nInfo, nRealizations);

    for r = 1:nRealizations

        rng(baseSeed + 800000 + 100000 * net + r);

        % Initialize near the stationary distribution of an OU process with
        % mean betaControl*S_z, unit decay rate and diffusion sigmaNoise.
        x = betaControl * S_z ...
            + sigmaNoise / sqrt(2) * randn(N,1);

        infoCounter = 0;

        for step = 1:Nt

            if step >= burnStep && mod(step - burnStep, infoStride) == 0
                infoCounter = infoCounter + 1;
                x_c0(:,infoCounter,r) = x;
            end

            if step == Nt
                break;
            end

            drift = -x + betaControl * S_z;
            x = x + drift * dt ...
                + sigmaNoise * sqrt(dt) * randn(N,1);
        end

        assert(infoCounter == nInfo, 'Unexpected structural-drive sample count.');
    end

    [I_c0(net), eta_c0(net), p_c0(net)] = ...
        analyze_standardized_cmi( ...
            x_c0, Sbin, Ssur, lag, infoThin, nBinsX, nBinsS);

    %% --------------------------------------------------------------------
    % Paired model differences
    % ---------------------------------------------------------------------

    delta_nonlin_linear(net) = I_nonlin(net) - I_linear(net);
    delta_linear_c0(net) = I_linear(net) - I_c0(net);

    %% --------------------------------------------------------------------
    % Per-network summary
    % ---------------------------------------------------------------------

    fprintf('\n');
    fprintf('Nonlinear: I = %.6f bits, eta = %.3f%%, p = %.4f\n', ...
        I_nonlin(net), 100 * eta_nonlin(net), p_nonlin(net));
    fprintf('Linear:    I = %.6f bits, eta = %.3f%%, p = %.4f\n', ...
        I_linear(net), 100 * eta_linear(net), p_linear(net));
    fprintf('C=0:       I = %.6f bits, eta = %.3f%%, p = %.4f\n', ...
        I_c0(net), 100 * eta_c0(net), p_c0(net));
    fprintf('Nonlinear - linear Delta I = %.6f bits\n', ...
        delta_nonlin_linear(net));
    fprintf('Linear - C=0 Delta I        = %.6f bits\n', ...
        delta_linear_c0(net));
end

%% ========================================================================
% Across-network summaries
% =========================================================================

% Standard deviations and standard errors across independently generated
% networks. The manuscript reports mean +/- SEM; both are saved below.
sd_I_nonlin = std(I_nonlin);
sd_I_linear = std(I_linear);
sd_I_c0 = std(I_c0);

sem_I_nonlin = sd_I_nonlin / sqrt(nNetworks);
sem_I_linear = sd_I_linear / sqrt(nNetworks);
sem_I_c0 = sd_I_c0 / sqrt(nNetworks);

sd_eta_nonlin = std(eta_nonlin);
sd_eta_linear = std(eta_linear);
sd_eta_c0 = std(eta_c0);

sem_eta_nonlin = sd_eta_nonlin / sqrt(nNetworks);
sem_eta_linear = sd_eta_linear / sqrt(nNetworks);
sem_eta_c0 = sd_eta_c0 / sqrt(nNetworks);

sd_delta_nonlin_linear = std(delta_nonlin_linear);
sd_delta_linear_c0 = std(delta_linear_c0);
sem_delta_nonlin_linear = sd_delta_nonlin_linear / sqrt(nNetworks);
sem_delta_linear_c0 = sd_delta_linear_c0 / sqrt(nNetworks);

fprintf('\n\n====================================================\n');
fprintf('FINAL MATCHED-CONTROL RESULTS (mean +/- SEM)\n');
fprintf('====================================================\n');

fprintf('\nNONLINEAR COUPLED\n');
fprintf('I = %.6f +/- %.6f bits\n', mean(I_nonlin), sem_I_nonlin);
fprintf('eta = %.3f +/- %.3f %%\n', ...
    100 * mean(eta_nonlin), 100 * sem_eta_nonlin);
fprintf('significant networks = %d/%d\n', ...
    sum(p_nonlin <= 0.05), nNetworks);

fprintf('\nMATCHED LINEAR COUPLED\n');
fprintf('I = %.6f +/- %.6f bits\n', mean(I_linear), sem_I_linear);
fprintf('eta = %.3f +/- %.3f %%\n', ...
    100 * mean(eta_linear), 100 * sem_eta_linear);
fprintf('significant networks = %d/%d\n', ...
    sum(p_linear <= 0.05), nNetworks);

fprintf('\nSTRUCTURAL-DRIVE CONTROL (C = 0)\n');
fprintf('I = %.6f +/- %.6f bits\n', mean(I_c0), sem_I_c0);
fprintf('eta = %.3f +/- %.3f %%\n', ...
    100 * mean(eta_c0), 100 * sem_eta_c0);
fprintf('significant networks = %d/%d\n', ...
    sum(p_c0 <= 0.05), nNetworks);

%% ------------------------------------------------------------------------
% Paired nonlinear vs linear comparison
% -------------------------------------------------------------------------

fprintf('\nNONLINEAR MINUS LINEAR\n');
fprintf('Delta I = %.6f +/- %.6f bits\n', ...
    mean(delta_nonlin_linear), sem_delta_nonlin_linear);
fprintf('positive networks = %d/%d\n', ...
    sum(delta_nonlin_linear > 0), nNetworks);

pSignNL = binomial_tail_two_sided( ...
    sum(delta_nonlin_linear > 0), ...
    sum(delta_nonlin_linear ~= 0));
fprintf('sign-test p = %.6g\n', pSignNL);

if exist('signrank', 'file') == 2
    pWilcoxonNL = signrank(I_nonlin, I_linear);
    fprintf('Wilcoxon p = %.6g\n', pWilcoxonNL);
else
    pWilcoxonNL = NaN;
    fprintf('Wilcoxon p = unavailable (signrank not found)\n');
end

%% ------------------------------------------------------------------------
% Paired linear vs structural-drive comparison
% -------------------------------------------------------------------------

fprintf('\nLINEAR MINUS C=0\n');
fprintf('Delta I = %.6f +/- %.6f bits\n', ...
    mean(delta_linear_c0), sem_delta_linear_c0);
fprintf('positive networks = %d/%d\n', ...
    sum(delta_linear_c0 > 0), nNetworks);

pSignLC = binomial_tail_two_sided( ...
    sum(delta_linear_c0 > 0), ...
    sum(delta_linear_c0 ~= 0));
fprintf('sign-test p = %.6g\n', pSignLC);

if exist('signrank', 'file') == 2
    pWilcoxonLC = signrank(I_linear, I_c0);
    fprintf('Wilcoxon p = %.6g\n', pWilcoxonLC);
else
    pWilcoxonLC = NaN;
    fprintf('Wilcoxon p = unavailable (signrank not found)\n');
end

%% ========================================================================
% Figures
% =========================================================================

figure('Name', 'Structural predictive information');
for net = 1:nNetworks
    plot([1 2 3], ...
        [I_c0(net), I_linear(net), I_nonlin(net)], ...
        'o-', 'LineWidth', 1);
    hold on;
end
xlim([0.7 3.3]);
xticks([1 2 3]);
xticklabels({'Structural drive', 'Linear coupled', 'Nonlinear coupled'});
ylabel('Conditional mutual information, I (bits)');
title('Structural predictive information across model classes');
box off;

figure('Name', 'Normalized structural predictive contribution');
for net = 1:nNetworks
    plot([1 2 3], ...
        100 * [eta_c0(net), eta_linear(net), eta_nonlin(net)], ...
        'o-', 'LineWidth', 1);
    hold on;
end
xlim([0.7 3.3]);
xticks([1 2 3]);
xticklabels({'Structural drive', 'Linear coupled', 'Nonlinear coupled'});
ylabel('Residual uncertainty explained, \eta (%)');
title('Normalized structural predictive contribution');
box off;

%% ========================================================================
% Save all numerical outputs needed to reproduce manuscript summaries
% =========================================================================

save('structure_dynamics_simulation_results.mat', ...
    'I_nonlin', 'eta_nonlin', 'p_nonlin', ...
    'I_linear', 'eta_linear', 'p_linear', ...
    'I_c0', 'eta_c0', 'p_c0', ...
    'delta_nonlin_linear', 'delta_linear_c0', ...
    'meanGainNetwork', 'g_all', ...
    'pSignNL', 'pWilcoxonNL', 'pSignLC', 'pWilcoxonLC', ...
    'sd_I_nonlin', 'sd_I_linear', 'sd_I_c0', ...
    'sem_I_nonlin', 'sem_I_linear', 'sem_I_c0', ...
    'sd_eta_nonlin', 'sd_eta_linear', 'sd_eta_c0', ...
    'sem_eta_nonlin', 'sem_eta_linear', 'sem_eta_c0', ...
    'sd_delta_nonlin_linear', 'sd_delta_linear_c0', ...
    'sem_delta_nonlin_linear', 'sem_delta_linear_c0', ...
    'baseSeed', 'N', 'nNetworks', 'nRealizations', ...
    'dt', 'T', 'burnIn', 'dtInfo', 'tau', 'infoThin', ...
    'sigmaNoise', 'targetEffectiveRadius', ...
    'fractionNegative', 'negativeStrength', 'lengthScale', ...
    'nBinsX', 'nBinsS', 'betaControl');

%% ========================================================================
% Local functions
% =========================================================================

function [I, eta, p] = analyze_standardized_cmi( ...
    X, Sbin, Ssur, lag, infoThin, nBinsX, nBinsS)
%ANALYZE_STANDARDIZED_CMI Estimate CMI after within-series z-scoring.
%
% Inputs
%   X         N x time x realization activity array.
%   Sbin      N x 1 structural labels for the observed network.
%   Ssur      N x nSur spatially constrained surrogate labels.
%   lag       Prediction lag in sampled time points.
%   infoThin  Thinning factor before pooling samples.
%   nBinsX    Number of quantile bins for present/future activity.
%   nBinsS    Number of structural bins.
%
% Outputs
%   I         I(S; X_future | X_present), in bits.
%   eta       I / H(X_future | X_present).
%   p         One-sided empirical surrogate p-value.

    [N, nTime, ~] = size(X);

    % Standardize independently within each region and realization.
    mu = mean(X, 2);
    Xc = X - mu;
    sd = std(X, 0, 2);
    sd(sd < eps) = 1;
    Xz = Xc ./ sd;

    validTimes = 1:infoThin:(nTime - lag);

    [Xp, Xf, regionID] = pool_samples(Xz, validTimes, lag);

    % Discretize pooled present and future activity by quantiles.
    XpBin = quantile_bin(Xp, nBinsX);
    XfBin = quantile_bin(Xf, nBinsX);

    % Keep region-specific state-transition counts so that the structural
    % labels can be reassigned cheaply for every spatial surrogate.
    regionCounts = build_region_counts( ...
        regionID, XfBin, XpBin, N, nBinsX);

    countsObserved = aggregate_by_structure( ...
        regionCounts, Sbin, nBinsS);
    I = cmi_from_counts(countsObserved);

    Hcond = conditional_entropy_from_region_counts(regionCounts);
    if Hcond > 0
        eta = I / Hcond;
    else
        eta = NaN;
    end

    nSur = size(Ssur, 2);
    nullValues = zeros(nSur, 1);

    for s = 1:nSur
        countsNull = aggregate_by_structure( ...
            regionCounts, Ssur(:,s), nBinsS);
        nullValues(s) = cmi_from_counts(countsNull);
    end

    % Add-one correction avoids zero permutation p-values.
    p = (1 + sum(nullValues >= I)) / (nSur + 1);
end


function [Xp, Xf, regionID] = pool_samples(X, validTimes, lag)
%POOL_SAMPLES Pool present/future samples across time and realizations.

    [N, ~, nRealizations] = size(X);
    nValid = numel(validTimes);
    totalSamples = N * nValid * nRealizations;

    Xp = zeros(totalSamples, 1);
    Xf = zeros(totalSamples, 1);
    regionID = zeros(totalSamples, 1);

    cursor = 1;

    for r = 1:nRealizations
        P = X(:, validTimes, r);
        F = X(:, validTimes + lag, r);

        nBlock = numel(P);
        inds = cursor:(cursor + nBlock - 1);

        Xp(inds) = P(:);
        Xf(inds) = F(:);
        regionID(inds) = repmat((1:N)', nValid, 1);

        cursor = cursor + nBlock;
    end
end


function bins = quantile_bin(x, nBins)
%QUANTILE_BIN Assign observations to approximately equal-frequency bins.
% Repeated quantile boundaries are separated by machine precision so that
% discretize() receives strictly increasing edges.

    x = x(:);
    edges = quantile(x, linspace(0, 1, nBins + 1));
    edges(1) = -Inf;
    edges(end) = Inf;

    for k = 2:length(edges)-1
        if edges(k) <= edges(k-1)
            edges(k) = edges(k-1) + eps(edges(k-1) + 1);
        end
    end

    bins = discretize(x, edges);
end


function countsR = build_region_counts(regionID, Y, X, N, nBinsX)
%BUILD_REGION_COUNTS Count future/present state-bin pairs for each region.

    countsR = zeros(N, nBinsX, nBinsX);

    for r = 1:N
        idx = (regionID == r);
        if ~any(idx)
            continue;
        end

        c = accumarray([Y(idx), X(idx)], 1, [nBinsX, nBinsX]);
        countsR(r,:,:) = c;
    end
end


function counts = aggregate_by_structure(regionCounts, Slabels, nBinsS)
%AGGREGATE_BY_STRUCTURE Pool region transition counts by structural label.

    [~, nY, nX] = size(regionCounts);
    counts = zeros(nBinsS, nY, nX);

    for s = 1:nBinsS
        idx = (Slabels == s);
        counts(s,:,:) = sum(regionCounts(idx,:,:), 1);
    end
end


function I = cmi_from_counts(counts)
%CMI_FROM_COUNTS Compute plug-in I(S;Y|X) from S-by-Y-by-X counts.

    Ntot = sum(counts(:));
    if Ntot <= 0
        I = NaN;
        return;
    end

    pSYX = counts / Ntot;
    pSX = squeeze(sum(pSYX, 2));
    pYX = squeeze(sum(pSYX, 1));
    pX = squeeze(sum(sum(pSYX, 1), 2));

    nS = size(pSYX, 1);
    nY = size(pSYX, 2);
    nX = size(pSYX, 3);

    I = 0;

    for s = 1:nS
        for y = 1:nY
            for x = 1:nX
                pJoint = pSYX(s,y,x);
                if pJoint <= 0
                    continue;
                end

                p_sx = pSX(s,x);
                p_yx = pYX(y,x);
                p_x = pX(x);

                if p_sx <= 0 || p_yx <= 0 || p_x <= 0
                    continue;
                end

                I = I + pJoint * log2((pJoint * p_x) / (p_sx * p_yx));
            end
        end
    end
end


function H = conditional_entropy_from_region_counts(regionCounts)
%CONDITIONAL_ENTROPY_FROM_REGION_COUNTS Compute H(Y|X), in bits.

    countsYX = squeeze(sum(regionCounts, 1));
    Ntot = sum(countsYX(:));

    if Ntot <= 0
        H = NaN;
        return;
    end

    pYX = countsYX / Ntot;
    pX = sum(pYX, 1);

    H = 0;

    for y = 1:size(pYX, 1)
        for x = 1:size(pYX, 2)
            pJoint = pYX(y,x);
            if pJoint <= 0 || pX(x) <= 0
                continue;
            end
            H = H - pJoint * log2(pJoint / pX(x));
        end
    end
end


function p = binomial_tail_two_sided(k, n)
%BINOMIAL_TAIL_TWO_SIDED Exact two-sided sign-test p-value under p = 0.5.

    if n == 0
        p = 1;
        return;
    end

    low = min(k, n-k);
    tail = 0;

    for j = 0:low
        tail = tail + nchoosek(n, j) * 0.5^n;
    end

    p = min(1, 2 * tail);
end
