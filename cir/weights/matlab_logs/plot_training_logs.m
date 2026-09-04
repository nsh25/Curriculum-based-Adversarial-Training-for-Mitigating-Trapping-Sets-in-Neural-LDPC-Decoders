%% plot_training_logs.m
% Plots RL-NMS / SAC-NMS / Adv-NMS training telemetry gathered by
% train_rl_sac_adv_logged.py. Run from this folder in MATLAB.
%
% Files: rlnms_log.mat, sacnms_log.mat, advnms_log.mat, training_logs_all.mat
% Each per-method file has:
%   step, reward|loss                         (per training step)
%   val_step, val_fer, val_ber                (per validation checkpoint)
%   wvn_mean, wvn_std, wcn_mean, wcn_std      (weight stats per checkpoint)
%   ebno_db, fer_curve, ber_curve             (final IS curve @ +/-7.5)
%   w_vn_final, w_cn_final [num_iter x E]      (final per-edge weights)
%   cn_idx, vn_idx, gamma, beta

clear; clc;
RL  = load('rlnms_log.mat');
SAC = load('sacnms_log.mat');
ADV = load('advnms_log.mat');
ALL = load('training_logs_all.mat');
methods = {RL, SAC, ADV};
names   = {'RL-NMS','SAC-NMS','Adv-NMS'};
cols    = lines(3);

%% 1) training signal (reward for ES, loss for Adv) over steps
figure('Name','Training signal','Color','w');
for i = 1:3
    M = methods{i};
    if isfield(M,'reward'), y = M.reward; ylab = 'reward'; else, y = M.loss; ylab = 'loss'; end
    subplot(1,3,i); plot(M.step, y, '-', 'Color', cols(i,:), 'LineWidth', 1.3);
    grid on; xlabel('training step'); ylabel(ylab); title(names{i});
end

%% 2) validation FER + BER over training
figure('Name','Val FER/BER over training','Color','w');
subplot(1,2,1);
for i = 1:3, M = methods{i}; semilogy(M.val_step, M.val_fer, '-o','Color',cols(i,:),'LineWidth',1.3); hold on; end
grid on; xlabel('training step'); ylabel('FER @4.5 dB (IS)'); legend(names); title('FER during training');
subplot(1,2,2);
for i = 1:3, M = methods{i}; semilogy(M.val_step, M.val_ber, '-s','Color',cols(i,:),'LineWidth',1.3); hold on; end
grid on; xlabel('training step'); ylabel('BER @4.5 dB (IS)'); legend(names); title('BER during training');

%% 3) final BER/FER vs Eb/N0 (with MS baseline)
figure('Name','Final BER/FER vs SNR','Color','w');
subplot(1,2,1);
semilogy(ALL.ebno_db, ALL.MS_fer, 'k--', 'LineWidth', 1.5); hold on;
for i = 1:3, M = methods{i}; semilogy(M.ebno_db, M.fer_curve, '-o','Color',cols(i,:),'LineWidth',1.4); end
grid on; xlabel('E_b/N_0 (dB)'); ylabel('FER'); legend(['MS' names]); title('Final FER @ clip \pm7.5');
subplot(1,2,2);
semilogy(ALL.ebno_db, ALL.MS_ber, 'k--', 'LineWidth', 1.5); hold on;
for i = 1:3, M = methods{i}; semilogy(M.ebno_db, M.ber_curve, '-s','Color',cols(i,:),'LineWidth',1.4); end
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER'); legend(['MS' names]); title('Final BER @ clip \pm7.5');

%% 4) decoder-weight statistics over training
figure('Name','Weight stats over training','Color','w');
subplot(1,2,1);
for i = 1:3, M = methods{i}; plot(M.val_step, M.wvn_mean, '-o','Color',cols(i,:),'LineWidth',1.3); hold on; end
grid on; xlabel('training step'); ylabel('mean(w_{vn})'); legend(names); title('VN-weight mean');
subplot(1,2,2);
for i = 1:3, M = methods{i}; plot(M.val_step, M.wvn_std, '-s','Color',cols(i,:),'LineWidth',1.3); hold on; end
grid on; xlabel('training step'); ylabel('std(w_{vn})'); legend(names); title('VN-weight spread');

%% 5) final per-edge weight distribution + per-iteration heatmap
figure('Name','Final decoder weights','Color','w');
subplot(2,2,1);
for i = 1:3, M = methods{i}; histogram(M.w_vn_final(:), 60, 'DisplayStyle','stairs','EdgeColor',cols(i,:),'LineWidth',1.3); hold on; end
grid on; xlabel('w_{vn}'); ylabel('count'); legend(names); title('VN weight histogram');
subplot(2,2,2);
for i = 1:3, M = methods{i}; histogram(M.w_cn_final(:), 60, 'DisplayStyle','stairs','EdgeColor',cols(i,:),'LineWidth',1.3); hold on; end
grid on; xlabel('w_{cn}'); ylabel('count'); legend(names); title('CN weight histogram');
subplot(2,2,3);
imagesc(ADV.w_vn_final); colorbar; xlabel('edge'); ylabel('iteration'); title('Adv-NMS w_{vn} [iter x edge]');
subplot(2,2,4);
imagesc(ADV.w_cn_final); colorbar; xlabel('edge'); ylabel('iteration'); title('Adv-NMS w_{cn} [iter x edge]');

disp('Plotted RL/SAC/Adv training logs.');
