%% plot_full_curve.m
% Full BER/FER curves: MC (waterfall) + IS (floor).
% Data: full_curve_mc_is.mat  (from mc_is_full_curve.py)
%
% Fix: MATLAB keeps orientation under logical indexing, so
%   [col; rowVec(mask)]  -> vertcat error.
% Always force columns with (:) *after* indexing.

clear; clc;
here = fileparts(mfilename('fullpath'));
matFile = fullfile(here, 'full_curve_mc_is.mat');
if ~isfile(matFile)
    error('Missing %s — run mc_is_full_curve.py first (cwd = this folder).', matFile);
end
D = load(matFile);

mcE = D.mc_ebno(:);
isE = D.is_ebno(:);
% Keep IS points strictly above the last MC Eb/N0 (drop overlap duplicate).
isKeep = isE > mcE(end) + 1e-9;
Efull = [mcE; isE(isKeep)];

cand = {'MS','NNMS','beatNNMS','IS_NMS','RL_warm','SAC_warm','Adv_warm'};
pretty = {'MS','NNMS','beat-NNMS','IS-NMS','RL-NMS (warm)','SAC-NMS (warm)','Adv-NMS (warm)'};

% Keep only decoders that exist and match SNR vector lengths.
names = {};
disp_names = {};
for i = 1:numel(cand)
    n = cand{i};
    need = { [n '_mc_fer'], [n '_is_fer'], [n '_mc_ber'], [n '_is_ber'] };
    ok = all(cellfun(@(f) isfield(D, f), need));
    if ~ok
        warning('Skipping %s (missing fields).', n);
        continue;
    end
    mcf = D.([n '_mc_fer'])(:);
    isf = D.([n '_is_fer'])(:);
    if numel(mcf) ~= numel(mcE) || numel(isf) ~= numel(isE)
        warning('Skipping %s (length mismatch: mc %d/%d, is %d/%d).', ...
            n, numel(mcf), numel(mcE), numel(isf), numel(isE));
        continue;
    end
    names{end+1} = n; %#ok<AGROW>
    disp_names{end+1} = pretty{i}; %#ok<AGROW>
end
if isempty(names)
    error('No usable decoder curves in %s', matFile);
end
cols = lines(numel(names));
% Distinct style + marker + growing size per curve, with OPEN markers, so the
% overlapping floor curves (SAC/Adv/RL-warm sit right on IS-NMS) stay visible.
ls  = {'-','-','--','-.',':','--','-.'};
mk  = {'o','s','^','d','v','>','p'};
msz = [6 6 7 8 9 11 13];

figure('Name','Full FER curve','Color','w'); hold on;
for i = 1:numel(names)
    n = names{i};
    mcf = D.([n '_mc_fer'])(:);
    isf = D.([n '_is_fer'])(:);
    FER = [mcf; isf(isKeep)];          % both columns after (:) above
    FER = FER(:);                      % belt-and-suspenders
    j = mod(i-1,numel(mk))+1;
    semilogy(Efull, max(FER, realmin), [ls{j} mk{j}], 'Color', cols(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', msz(j), 'DisplayName', disp_names{i});
end
set(gca,'YScale','log'); grid on; box on;
xlabel('E_b/N_0 (dB)'); ylabel('FER');
title(sprintf('FER: MC (\\leq%.1f dB) + IS (>%.1f dB)', mcE(end), mcE(end)));
legend('Location','southwest');
xline(mcE(end), 'k:', 'MC \rightarrow IS');

figure('Name','Full BER curve','Color','w'); hold on;
for i = 1:numel(names)
    n = names{i};
    mcb = D.([n '_mc_ber'])(:);
    isb = D.([n '_is_ber'])(:);
    BER = [mcb; isb(isKeep)];
    BER = BER(:);
    j = mod(i-1,numel(mk))+1;
    semilogy(Efull, max(BER, realmin), [ls{j} mk{j}], 'Color', cols(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', msz(j), 'DisplayName', disp_names{i});
end
set(gca,'YScale','log'); grid on; box on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title(sprintf('BER: MC (\\leq%.1f dB) + IS (>%.1f dB)', mcE(end), mcE(end)));
legend('Location','southwest');
xline(mcE(end), 'k:', 'MC \rightarrow IS');

% Overlap sanity: last MC SNR vs first IS SNR (usually the shared edge point)
fprintf('\nOverlap FER (MC last @ %.2f dB | IS first @ %.2f dB):\n', mcE(end), isE(1));
for i = 1:numel(names)
    n = names{i};
    fprintf('  %-10s  %.3e | %.3e\n', n, D.([n '_mc_fer'])(end), D.([n '_is_fer'])(1));
end
disp('Plotted full MC+IS BER/FER curves.');
