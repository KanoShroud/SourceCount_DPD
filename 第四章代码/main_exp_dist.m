%% main_exp_dist.m  距离扫描控制变量实验（简化版：仅生成IQ）
%
% 实验 4B2: N=2, SNR=0dB, 方位30°/150°+随机旋转, 距离扫描
% 实验 4B3: N=3, SNR=0dB, 方位30°/150°/270°+随机旋转, 距离扫描
%
% 距离范围: 100:100:1000 m，每个距离 2000 样本，最小源站距 200m
%
clc; clear; close all;
tic
global Txobj Rxobj

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fileparts(script_dir));
runtime = gate0_runtime('chapter4', mfilename);

if isempty(gcp('nocreate')), parpool('local'); end

%% ── 系统参数 ──
fc          = 5800e6;
arfa_V      = 0.25;
symbolRate  = 10e6;
BW_actual   = symbolRate * (1 + arfa_V * 1.2);
fs          = 100e6;
len         = 2^12;
vc          = 299792458;
N_sub       = 19;
max_src     = 3;

N_power_dBm = -90;
N_power_W   = 10^((N_power_dBm - 30) / 10);

R_rcv = 500; N_rx = 4;
angles_rx = (0:N_rx-1) * 2*pi/N_rx;
Rxobj.Num = 4; Rxobj.rxId_V = ["001";"002";"003";"004"];
Rxobj.node_pos = [R_rcv*cos(angles_rx)', R_rcv*sin(angles_rx)', zeros(N_rx,1)];
Rxobj.sample_rate = repmat(fs, N_rx, 1); Rxobj.freq_rf = repmat(fc, N_rx, 1);
rcvPos = Rxobj.node_pos(:,1:2); rcv_num = N_rx;

Txobj.Num=1; Txobj.txId_V="drone001"; Txobj.freqC_V=fc;
Txobj.modType_V="BPSK"; Txobj.multiplexingType_V="NONE";
Txobj.shapingType_V="RootRaisedCos"; Txobj.arfa_V=arfa_V;
Txobj.symbolRate_V=symbolRate; Txobj.modDepth_V=1.0;
Txobj.contPhase_V="cont"; Txobj.nodePos_V=[0,0,0];
Txobj.Gt=0; Txobj.antennaType_V="AntennaX"; Txobj.antennaDeg_V=[0,0];
Txobj.transmitPower_V=1.0; Txobj.txTime_V="2020-06-06 00:00:00:000000";
Txobj.txDuration_V=0; Txobj.Bw_V=0;

dataLen = calc_dataLen(len, fs, symbolRate);

%% ── 实验配置 ──
dist_range_m      = 100:100:1000;
if runtime.is_smoke
    n_trials_per_dist = 1;
else
    n_trials_per_dist = 2000;
end
target_snr_dB     = 0;
target_snr_lin    = 10^(target_snr_dB / 10);
src_azimuths      = [30, 150, 330];
min_dist_src2rcv  = 200;    % 最小源站距离 (m)
exp_list          = {'4B2', '4B3'};
n_src_list        = [2, 3];
P_noise_inband    = N_power_W * (BW_actual / fs);

edge=2000; lamda=50; B_win=10e6; B_step=5e6; thresh=0.3;
sub_f_lo = ((0:N_sub-1)*B_step - fs/2);
sub_f_hi = sub_f_lo + B_win;

fprintf('距离: %d:%d:%d m, SNR=%ddB, 每个 %d 样本\n', ...
    dist_range_m(1), dist_range_m(2)-dist_range_m(1), dist_range_m(end), ...
    target_snr_dB, n_trials_per_dist);

%% ── 实验循环 ──
for exp_idx = 1:length(exp_list)
    exp_name = exp_list{exp_idx};
    n_src    = n_src_list(exp_idx);
    fprintf('\n========== %s (N=%d) ==========\n', exp_name, n_src);

    fc_off  = zeros(1, n_src);
    N_total = length(dist_range_m) * n_trials_per_dist;

    src_count_all    = zeros(N_total, 1, 'int32');
    avg_snr_all      = zeros(N_total, 1, 'single');
    fc_offset_all    = zeros(N_total, max_src, 'single');
    Pt_W_all         = zeros(N_total, max_src, 'single');
    src_pos_all      = zeros(N_total, max_src, 2, 'single');
    sig_rcv_real_all = zeros(N_total, rcv_num, len, 'single');
    sig_rcv_imag_all = zeros(N_total, rcv_num, len, 'single');
    dist_param_all   = zeros(N_total, 1, 'single');

    t_start = tic; trial_global = 0;

    for dist_idx = 1:length(dist_range_m)
        current_dist = dist_range_m(dist_idx);

        for ti = 1:n_trials_per_dist
            trial_global = trial_global + 1;

            % ── 随机旋转 + 最小源站距约束 ──
            while true
                rot_angle = 2*pi * rand();
                src_pos = zeros(n_src, 2);
                for s = 1:n_src
                    az_rad = deg2rad(src_azimuths(s)) + rot_angle;
                    src_pos(s,:) = [current_dist*cos(az_rad), current_dist*sin(az_rad)];
                end
                % 检查源到站最小距离
                d_min = inf;
                for s = 1:n_src
                    for m = 1:rcv_num
                        d_min = min(d_min, norm(src_pos(s,:) - rcvPos(m,:)));
                    end
                end
                if d_min >= min_dist_src2rcv, break; end
            end

            % ── 路径损耗 + 发射功率 ──
            PL_dB_mat = zeros(n_src, rcv_num);
            for s = 1:n_src
                for m = 1:rcv_num
                    PL_dB_mat(s,m) = PL_free(fc, norm(src_pos(s,:)-rcvPos(m,:)), 0, 0);
                end
            end
            PL_avg_dB = mean(PL_dB_mat, 2);
            Pt_W = zeros(1, n_src);
            for s = 1:n_src
                Pt_W(s) = target_snr_lin * P_noise_inband * 10^(PL_avg_dB(s)/10);
            end

            % ── IQ 信号生成 ──
            sig_pool = zeros(n_src, len);
            for s = 1:n_src
                [~, tmp] = Gen_basesig(dataLen, fs, Txobj.txId_V);
                tmp = tmp(1:len);
                sig_pool(s,:) = tmp / sqrt(mean(abs(tmp).^2));
            end

            sig_rcv_accum = zeros(rcv_num, len);
            t_vec = (0:len-1) / fs;
            snr_actual = zeros(n_src, rcv_num);
            for s = 1:n_src
                baseSig = sig_pool(s,:) .* exp(1j*2*pi*fc_off(s)*t_vec);
                baseSig = baseSig / sqrt(mean(abs(baseSig).^2));
                for m = 1:rcv_num
                    tau_m = norm(src_pos(s,:)-rcvPos(m,:)) / vc;
                    sig_del = apply_delay_fd(baseSig, tau_m, fs);
                    Pr_W = Pt_W(s) * 10^(-PL_dB_mat(s,m)/10);
                    sig_rcv_accum(m,:) = sig_rcv_accum(m,:) + sqrt(Pr_W)*sig_del;
                    snr_actual(s,m) = 10*log10(Pr_W / P_noise_inband);
                end
            end

            sig_rcv = zeros(rcv_num, len);
            for m = 1:rcv_num
                noise = sqrt(N_power_W/2) * (randn(1,len) + 1j*randn(1,len));
                sig_rcv(m,:) = sig_rcv_accum(m,:) + noise;
            end

            sig_rcv_real_all(trial_global,:,:) = single(real(sig_rcv));
            sig_rcv_imag_all(trial_global,:,:) = single(imag(sig_rcv));
            src_count_all(trial_global) = int32(n_src);
            fc_offset_all(trial_global,1:n_src) = single(fc_off(1:n_src));
            Pt_W_all(trial_global,1:n_src) = single(Pt_W(1:n_src));
            src_pos_all(trial_global,1:n_src,:) = single(src_pos);
            dist_param_all(trial_global) = single(current_dist);
            avg_snr_all(trial_global) = single(mean(mean(snr_actual,2)));

            if mod(trial_global, 500) == 0
                elapsed = toc(t_start);
                fprintf('[%s] %d/%d (%.0f%%) | %.0fs | ETA %.0fs\n', ...
                    exp_name, trial_global, N_total, 100*trial_global/N_total, ...
                    elapsed, elapsed/trial_global*(N_total-trial_global));
            end
        end
    end

    fprintf('\n===== %s 完毕 (%d样本, %.1fs) =====\n', exp_name, N_total, toc(t_start));

    save_file = fullfile(runtime.data_dir, sprintf('exp_dist_%s.mat', exp_name));
    rcv_pos_val=single(rcvPos); BW_actual_val=single(BW_actual);
    N_power_W_val=single(N_power_W); N_power_dBm_val=single(N_power_dBm);
    exp_n_src_val=int32(n_src); exp_dist_range_val=single(dist_range_m);
    exp_n_per_dist_val=int32(n_trials_per_dist); exp_snr_dB_val=single(target_snr_dB);
    exp_azimuths_val=single(src_azimuths(1:n_src));
    N_sub_val=int32(N_sub); max_src_val=int32(max_src);
    edge_val=single(edge); lamda_val=single(lamda);
    B_win_val=single(B_win); B_step_val=single(B_step);
    fs_val=single(fs); symbolRate_val=single(symbolRate);
    sub_f_lo_val=single(sub_f_lo); sub_f_hi_val=single(sub_f_hi);
    thresh_val=single(thresh); num_count_classes=int32(max_src+1);

    save(save_file, ...
        'src_count_all', 'avg_snr_all', 'fc_offset_all', 'Pt_W_all', 'src_pos_all', ...
        'sig_rcv_real_all', 'sig_rcv_imag_all', 'dist_param_all', ...
        'rcv_pos_val', 'BW_actual_val', 'N_power_W_val', 'N_power_dBm_val', ...
        'exp_n_src_val', 'exp_dist_range_val', 'exp_n_per_dist_val', ...
        'exp_snr_dB_val', 'exp_azimuths_val', ...
        'N_sub_val', 'max_src_val', 'edge_val', 'lamda_val', ...
        'B_win_val', 'B_step_val', 'fs_val', 'symbolRate_val', ...
        'sub_f_lo_val', 'sub_f_hi_val', 'thresh_val', 'num_count_classes', ...
        '-v7.3');
    fprintf('已保存: %s (%.1f GB)\n', save_file, dir(save_file).bytes/1e9);

    clear src_count_all avg_snr_all fc_offset_all Pt_W_all src_pos_all ...
          sig_rcv_real_all sig_rcv_imag_all dist_param_all;
end
fprintf('\n全部完成！%.1f 秒\n', toc);
