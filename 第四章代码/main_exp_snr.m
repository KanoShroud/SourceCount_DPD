%% main_exp_snr.m  SNR 扫描控制变量实验（简化版：仅生成IQ）
%
% 实验 4A2: N=2, 距离800m, 方位30°/150°固定, 同频
% 实验 4A3: N=3, 距离800m, 方位30°/150°/270°固定, 同频
%
% SNR 范围: -10:2:10 dB，每个 SNR: 2000 样本
% DPD 由 Python gen_exp_data.py 计算
%
clc; clear; close all;
tic
global Txobj Rxobj

if isempty(gcp('nocreate')), parpool('local'); end

%% ── 系统参数 ──
fc          = 5800e6;
arfa_V      = 0.25;
symbolRate  = 10e6;
BW_actual   = symbolRate * (1 + arfa_V * 1.2);   % 13MHz
fs          = 100e6;
len         = 2^12;
vc          = 299792458;
N_sub       = 19;
max_src     = 3;

%% ── 噪声参数 ──
N_power_dBm = -90;
N_power_W   = 10^((N_power_dBm - 30) / 10);

%% ── 接收站 ──
R_rcv  = 500; N_rx = 4;
angles_rx = (0:N_rx-1) * 2*pi/N_rx;
Rxobj.Num = 4;
Rxobj.rxId_V = ["001";"002";"003";"004"];
Rxobj.node_pos = [R_rcv*cos(angles_rx)', R_rcv*sin(angles_rx)', zeros(N_rx,1)];
Rxobj.sample_rate = repmat(fs, N_rx, 1);
Rxobj.freq_rf = repmat(fc, N_rx, 1);
rcvPos  = Rxobj.node_pos(:, 1:2);
rcv_num = N_rx;

%% ── 发射机模板 ──
Txobj.Num = 1; Txobj.txId_V = "drone001"; Txobj.freqC_V = fc;
Txobj.modType_V = "BPSK"; Txobj.multiplexingType_V = "NONE";
Txobj.shapingType_V = "RootRaisedCos"; Txobj.arfa_V = arfa_V;
Txobj.symbolRate_V = symbolRate; Txobj.modDepth_V = 1.0;
Txobj.contPhase_V = "cont"; Txobj.nodePos_V = [0,0,0];
Txobj.Gt = 0; Txobj.antennaType_V = "AntennaX";
Txobj.antennaDeg_V = [0,0]; Txobj.transmitPower_V = 1.0;
Txobj.txTime_V = "2020-06-06 00:00:00:000000";
Txobj.txDuration_V = 0; Txobj.Bw_V = 0;

dataLen = calc_dataLen(len, fs, symbolRate);

%% ── 实验配置 ──
snr_range_dB     = -10:2:10;
n_trials_per_snr = 5000;
src_distance     = 800;
src_azimuths     = [30, 150, 330];
exp_list         = {'4A2', '4A3'};
n_src_list       = [2, 3];
P_noise_inband   = N_power_W * (BW_actual / fs);

%% ── Python 可读参数 ──
edge = 2000; lamda = 50;
B_win = 10e6; B_step = 5e6;
sub_f_lo = ((0:N_sub-1)*B_step - fs/2);
sub_f_hi = sub_f_lo + B_win;
thresh = 0.3;

fprintf('SNR 范围: [%d, %d] dB, 每个 %d 样本, 距离 %dm\n', ...
    snr_range_dB(1), snr_range_dB(end), n_trials_per_snr, src_distance);

%% ── 实验循环 ──
for exp_idx = 1:length(exp_list)
    exp_name = exp_list{exp_idx};
    n_src    = n_src_list(exp_idx);
    fprintf('\n========== %s (N=%d) ==========\n', exp_name, n_src);

    fc_off = zeros(1, n_src);
    N_total = length(snr_range_dB) * n_trials_per_snr;

    % 预分配
    src_count_all    = zeros(N_total, 1, 'int32');
    avg_snr_all      = zeros(N_total, 1, 'single');
    fc_offset_all    = zeros(N_total, max_src, 'single');
    Pt_W_all         = zeros(N_total, max_src, 'single');
    src_pos_all      = zeros(N_total, max_src, 2, 'single');
    sig_rcv_real_all = zeros(N_total, rcv_num, len, 'single');
    sig_rcv_imag_all = zeros(N_total, rcv_num, len, 'single');
    snr_param_all    = zeros(N_total, 1, 'single');

    t_start = tic;
    trial_global = 0;

    for snr_idx = 1:length(snr_range_dB)
        target_snr_dB  = snr_range_dB(snr_idx);
        target_snr_lin = 10^(target_snr_dB / 10);

        for ti = 1:n_trials_per_snr
            trial_global = trial_global + 1;

            % ── 固定方位（无旋转，纯控制变量）──
            src_pos = zeros(n_src, 2);
            for s = 1:n_src
                az_rad = deg2rad(src_azimuths(s));
                src_pos(s,:) = [src_distance * cos(az_rad), src_distance * sin(az_rad)];
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

            % ── 生成基带信号 ──
            sig_pool = zeros(n_src, len);
            for s = 1:n_src
                [~, tmp] = Gen_basesig(dataLen, fs, Txobj.txId_V);
                tmp = tmp(1:len);
                sig_pool(s,:) = tmp / sqrt(mean(abs(tmp).^2));
            end

            % ── 构造接收信号 ──
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

            % ── 加噪声 ──
            sig_rcv = zeros(rcv_num, len);
            for m = 1:rcv_num
                noise = sqrt(N_power_W/2) * (randn(1,len) + 1j*randn(1,len));
                sig_rcv(m,:) = sig_rcv_accum(m,:) + noise;
            end

            % ── 写入 ──
            sig_rcv_real_all(trial_global,:,:) = single(real(sig_rcv));
            sig_rcv_imag_all(trial_global,:,:) = single(imag(sig_rcv));
            src_count_all(trial_global)           = int32(n_src);
            fc_offset_all(trial_global, 1:n_src)  = single(fc_off(1:n_src));
            Pt_W_all(trial_global, 1:n_src)       = single(Pt_W(1:n_src));
            src_pos_all(trial_global, 1:n_src, :) = single(src_pos);
            snr_param_all(trial_global)            = single(target_snr_dB);
            avg_snr_all(trial_global)              = single(mean(mean(snr_actual,2)));

            if mod(trial_global, 500) == 0
                elapsed = toc(t_start);
                eta = elapsed/trial_global*(N_total-trial_global);
                fprintf('[%s] %d/%d (%.0f%%) | %.0fs | ETA %.0fs\n', ...
                    exp_name, trial_global, N_total, 100*trial_global/N_total, elapsed, eta);
            end
        end
    end

    fprintf('\n===== %s 完毕 (%d样本, %.1fs) =====\n', exp_name, N_total, toc(t_start));

    % ── 保存 ──
    save_file = sprintf('exp_snr_%s.mat', exp_name);
    rcv_pos_val = single(rcvPos); BW_actual_val = single(BW_actual);
    N_power_W_val = single(N_power_W); N_power_dBm_val = single(N_power_dBm);
    exp_n_src_val = int32(n_src); exp_snr_range_val = single(snr_range_dB);
    exp_n_per_snr_val = int32(n_trials_per_snr);
    exp_distance_val = single(src_distance); exp_azimuths_val = single(src_azimuths(1:n_src));
    N_sub_val = int32(N_sub); max_src_val = int32(max_src);
    edge_val = single(edge); lamda_val = single(lamda);
    B_win_val = single(B_win); B_step_val = single(B_step);
    fs_val = single(fs); symbolRate_val = single(symbolRate);
    sub_f_lo_val = single(sub_f_lo); sub_f_hi_val = single(sub_f_hi);
    thresh_val = single(thresh); num_count_classes = int32(max_src+1);

    save(save_file, ...
        'src_count_all', 'avg_snr_all', 'fc_offset_all', 'Pt_W_all', 'src_pos_all', ...
        'sig_rcv_real_all', 'sig_rcv_imag_all', 'snr_param_all', ...
        'rcv_pos_val', 'BW_actual_val', 'N_power_W_val', 'N_power_dBm_val', ...
        'exp_n_src_val', 'exp_snr_range_val', 'exp_n_per_snr_val', ...
        'exp_distance_val', 'exp_azimuths_val', ...
        'N_sub_val', 'max_src_val', 'edge_val', 'lamda_val', ...
        'B_win_val', 'B_step_val', 'fs_val', 'symbolRate_val', ...
        'sub_f_lo_val', 'sub_f_hi_val', 'thresh_val', 'num_count_classes', ...
        '-v7.3');
    fprintf('已保存: %s (%.1f GB)\n', save_file, dir(save_file).bytes/1e9);

    clear src_count_all avg_snr_all fc_offset_all Pt_W_all src_pos_all ...
          sig_rcv_real_all sig_rcv_imag_all snr_param_all;
end

fprintf('\n全部完成！%.1f 秒\n', toc);