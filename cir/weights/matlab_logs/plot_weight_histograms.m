%% plot_weight_histograms.m
% Overlay + per-method histograms of VN/CN decoder weights for the same
% methods as plot_full_curve.m.
%
% Data: weights_for_hist.mat  (from export_weights_hist_mat.py next door,
% or regenerated with that script). Fields: <name>_w_vn, <name>_w_cn,
% each [num_iter x E].

clear; clc;
here = fileparts(mfilename('fullpath'));
matFile = fullfile(here, 'weights_for_hist.mat');
if ~isfile(matFile)
    error(['Missing %s — run export_weights_hist_mat.py from the ' ...
           '1280_R0.50 folder first.'], matFile);
end
D = load(matFile);

cand = {'NNMS','beatNNMS','IS_NMS','RL_warm','SAC_warm','Adv_warm'};
pretty = {'NNMS','beat-NNMS','IS-NMS','RL-NMS (warm)','SAC-NMS (warm)','Adv-NMS (warm)'};

names = {};
disp_names = {};
Wvn = {};
Wcn = {};
for i = 1:numel(cand)
    n = cand{i};
    fv = [n '_w_vn']; fc = [n '_w_cn'];
    if ~isfield(D, fv) || ~isfield(D, fc)
        warning('Skipping %s (missing weight fields).', n);
        continue;
    end
    names{end+1} = n; %#ok<AGROW>
    disp_names{end+1} = pretty{i}; %#ok<AGROW>
    Wvn{end+1} = D.(fv)(:); %#ok<AGROW>
    Wcn{end+1} = D.(fc)(:); %#ok<AGROW>
end
if isempty(names)
    error('No weight fields found in %s', matFile);
end
cols = lines(numel(names));
nbins = 60;

%% 1) Overlay stair histograms (all methods)
figure('Name','Weight histograms (overlay)','Color','w');
subplot(1,2,1); hold on;
for i = 1:numel(names)
    histogram(Wvn{i}, nbins, 'DisplayStyle','stairs', ...
        'EdgeColor', cols(i,:), 'LineWidth', 1.4, 'DisplayName', disp_names{i});
end
xline(1, 'k:', 'MS (w=1)', 'HandleVisibility','off');
grid on; box on; xlabel('w_{vn}'); ylabel('count');
title('VN weights'); legend('Location','best');

subplot(1,2,2); hold on;
for i = 1:numel(names)
    histogram(Wcn{i}, nbins, 'DisplayStyle','stairs', ...
        'EdgeColor', cols(i,:), 'LineWidth', 1.4, 'DisplayName', disp_names{i});
end
xline(1, 'k:', 'MS (w=1)', 'HandleVisibility','off');
grid on; box on; xlabel('w_{cn}'); ylabel('count');
title('CN weights'); legend('Location','best');

%% 2) Per-method filled histograms (shared bin edges)
allv = vertcat(Wvn{:});
allc = vertcat(Wcn{:});
edges_v = linspace(min(allv), max(allv), nbins+1);
edges_c = linspace(min(allc), max(allc), nbins+1);

figure('Name','Weight histograms (per method)','Color','w');
nM = numel(names);
for i = 1:nM
    subplot(nM, 2, 2*i-1);
    histogram(Wvn{i}, edges_v, 'FaceColor', cols(i,:), 'EdgeColor', 'none');
    xline(1, 'k:');
    grid on; box on; ylabel('count');
    title(sprintf('%s  w_{vn}  mean=%.3f std=%.3f', ...
        disp_names{i}, mean(Wvn{i}), std(Wvn{i})));
    if i == nM, xlabel('w_{vn}'); end

    subplot(nM, 2, 2*i);
    histogram(Wcn{i}, edges_c, 'FaceColor', cols(i,:), 'EdgeColor', 'none');
    xline(1, 'k:');
    grid on; box on;
    title(sprintf('%s  w_{cn}  mean=%.3f std=%.3f', ...
        disp_names{i}, mean(Wcn{i}), std(Wcn{i})));
    if i == nM, xlabel('w_{cn}'); end
end

%% 3) Stats table in console
fprintf('\n%-16s  %8s %8s %8s %8s   %8s %8s %8s %8s\n', ...
    'method', 'vn_min','vn_mean','vn_std','vn_max', ...
    'cn_min','cn_mean','cn_std','cn_max');
for i = 1:numel(names)
    v = Wvn{i}; c = Wcn{i};
    fprintf('%-16s  %8.3f %8.3f %8.3f %8.3f   %8.3f %8.3f %8.3f %8.3f\n', ...
        disp_names{i}, min(v), mean(v), std(v), max(v), ...
        min(c), mean(c), std(c), max(c));
end
disp('Plotted weight histograms.');
