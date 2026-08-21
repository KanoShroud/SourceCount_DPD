%% main_exp_bw.m  带宽扫描控制变量实验
%
% 实验 4D: N=2, SNR=0dB, 距离800m, 方位30°/150°固定, 同频
%          源1固定 symbolRate=10MHz, 源2 symbolRate 从 2:2:20 MHz 扫描
%
% 每个带宽点: 2000 样本
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
Txobj.symbolRate_V=10e6; Txobj.modDepth_V=1.0;
Txobj.contPhase_V="cont"; Txobj.nodePos_V=[0,0,0];
Txobj.Gt=0; Txobj.antennaType_V="AntennaX"; Txobj.antennaDeg_V=[0,0];
Txobj.transmitPower_V=1.0; Txobj.txTime_V="2020-06-06 00:00:00:000000";
Txobj.txDuration_V=0; Txobj.Bw_V=0;

%% ── 实验配置 ──
symbolRate_src1   = 10e6;                     % 源1 固定 10MHz
bw_range_MHz      = 2:2:20;                   % 源2 symbolRate 扫描 (MHz)
if runtime.is_smoke
    n_trials_per_bw = 1;
else
    n_trials_per_bw = 2000;
end
src_distance      = 800;
src_azimuths      = [30, 150];
target_snr_dB     = 0;
target_snr_lin    = 10^(target_snr_dB / 10);
n_src             = 2;

BW_src1 = symbolRate_src1 * (1 + arfa_V * 1.2);
P_noise_inband_src1 = N_power_W * (BW_src1 / fs);

edge=2000; lamda=50; B_win=10e6; B_step=5e6; thresh=0.3;
sub_f_lo = ((0:N_sub-1)*B_step - fs/2);
sub_f_hi = sub_f_lo + B_win;

fprintf('带宽扫描: 源2 symbolRate = %d:%d:%d MHz\n', ...
    bw_range_MHz(1), bw_range_MHz(2)-bw_range_MHz(1), bw_range_MHz(end));
fprintf('源1: 固定 %d MHz, SNR=%ddB, 距离=%dm, 每点 %d 样本\n', ...
    symbolRate_src1/1e6, target_snr_dB, src_distance, n_trials_per_bw);

%% ── 主循环 ──
N_total = length(bw_range_MHz) * n_trials_per_bw;

src_count_all    = zeros(N_total, 1, 'int32');
avg_snr_all      = zeros(N_total, 1, 'single');
fc_offset_all    = zeros(N_total, max_src, 'single');
Pt_W_all         = zeros(N_total, max_src, 'single');
src_pos_all      = zeros(N_total, max_src, 2, 'single');
sig_rcv_real_all = zeros(N_total, rcv_num, len, 'single');
sig_rcv_imag_all = zeros(N_total, rcv_num, len, 'single');
bw_param_all     = zeros(N_total, 1, 'single');
BW_actual_all    = zeros(N_total, max_src, 'single');

fc_off = [0, 0];

t_start = tic; trial_global = 0;

for bw_idx = 1:length(bw_range_MHz)
    symbolRate_src2 = bw_range_MHz(bw_idx) * 1e6;
    BW_src2 = symbolRate_src2 * (1 + arfa_V * 1.2);

    % 每源带内噪声功率（SNR 按各源带宽计算）
    P_noise_src1 = N_power_W * (BW_src1 / fs);
    P_noise_src2 = N_power_W * (BW_src2 / fs);

    % 每源 dataLen
    dataLen_src1 = calc_dataLen(len, fs, symbolRate_src1);
    dataLen_src2 = calc_dataLen(len, fs, symbolRate_src2);

    fprintf('\n--- 源2 symbolRate = %d MHz (BW=%.1f MHz) ---\n', ...
        bw_range_MHz(bw_idx), BW_src2/1e6);

    for ti = 1:n_trials_per_bw
        trial_global = trial_global + 1;

        % ── 固定方位 ──
        src_pos = zeros(n_src, 2);
        for s = 1:n_src
            az_rad = deg2rad(src_azimuths(s));
            src_pos(s,:) = [src_distance * cos(az_rad), src_distance * sin(az_rad)];
        end

        % ── 路径损耗 ──
        PL_dB_mat = zeros(n_src, rcv_num);
        for s = 1:n_src
            for m = 1:rcv_num
                PL_dB_mat(s,m) = PL_free(fc, norm(src_pos(s,:)-rcvPos(m,:)), 0, 0);
            end
        end
        PL_avg_dB = mean(PL_dB_mat, 2);

        % ── 发射功率（按各源带内噪声功率保证 SNR）──
        Pt_W = zeros(1, n_src);
        Pt_W(1) = target_snr_lin * P_noise_src1 * 10^(PL_avg_dB(1)/10);
        Pt_W(2) = target_snr_lin * P_noise_src2 * 10^(PL_avg_dB(2)/10);

        % ── 生成基带信号（不同 symbolRate）──
        sig_pool = zeros(n_src, len);

        % 源1: 固定 symbolRate
        Txobj.symbolRate_V = symbolRate_src1;
        [~, tmp] = Gen_basesig(dataLen_src1, fs, Txobj.txId_V);
        tmp = tmp(1:len);
        sig_pool(1,:) = tmp / sqrt(mean(abs(tmp).^2));

        % 源2: 可变 symbolRate
        Txobj.symbolRate_V = symbolRate_src2;
        [~, tmp] = Gen_basesig(dataLen_src2, fs, Txobj.txId_V);
        tmp = tmp(1:len);
        sig_pool(2,:) = tmp / sqrt(mean(abs(tmp).^2));

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
                if s == 1
                    snr_actual(s,m) = 10*log10(Pr_W / P_noise_src1);
                else
                    snr_actual(s,m) = 10*log10(Pr_W / P_noise_src2);
                end
            end
        end

        sig_rcv = zeros(rcv_num, len);
        for m = 1:rcv_num
            noise = sqrt(N_power_W/2) * (randn(1,len) + 1j*randn(1,len));
            sig_rcv(m,:) = sig_rcv_accum(m,:) + noise;
        end

        % ── 写入 ──
        sig_rcv_real_all(trial_global,:,:) = single(real(sig_rcv));
        sig_rcv_imag_all(trial_global,:,:) = single(imag(sig_rcv));
        src_count_all(trial_global) = int32(n_src);
        fc_offset_all(trial_global,1:n_src) = single(fc_off);
        Pt_W_all(trial_global,1:n_src) = single(Pt_W);
        src_pos_all(trial_global,1:n_src,:) = single(src_pos);
        bw_param_all(trial_global) = single(bw_range_MHz(bw_idx));
        avg_snr_all(trial_global) = single(mean(mean(snr_actual,2)));
        BW_actual_all(trial_global,1) = single(BW_src1);
        BW_actual_all(trial_global,2) = single(BW_src2);

        if mod(trial_global, 500) == 0
            elapsed = toc(t_start);
            fprintf('[4D] %d/%d (%.0f%%) | %.0fs | ETA %.0fs\n', ...
                trial_global, N_total, 100*trial_global/N_total, ...
                elapsed, elapsed/trial_global*(N_total-trial_global));
        end
    end
end

% 恢复默认
Txobj.symbolRate_V = symbolRate_src1;

fprintf('\n===== 4D 完毕 (%d样本, %.1fs) =====\n', trial_global, toc(t_start));

%% ── 保存 ──
save_file = fullfile(runtime.data_dir, 'exp_bw_4D.mat');
rcv_pos_val=single(rcvPos); N_power_W_val=single(N_power_W);
N_power_dBm_val=single(N_power_dBm);
exp_n_src_val=int32(n_src); exp_bw_range_val=single(bw_range_MHz);
exp_n_per_bw_val=int32(n_trials_per_bw);
exp_snr_dB_val=single(target_snr_dB); exp_distance_val=single(src_distance);
exp_azimuths_val=single(src_azimuths);
symbolRate_src1_val=single(symbolRate_src1);
N_sub_val=int32(N_sub); max_src_val=int32(max_src);
edge_val=single(edge); lamda_val=single(lamda);
B_win_val=single(B_win); B_step_val=single(B_step);
fs_val=single(fs); symbolRate_val=single(symbolRate_src1);
sub_f_lo_val=single(sub_f_lo); sub_f_hi_val=single(sub_f_hi);
thresh_val=single(thresh); num_count_classes=int32(max_src+1);

save(save_file, ...
    'src_count_all', 'avg_snr_all', 'fc_offset_all', 'Pt_W_all', 'src_pos_all', ...
    'sig_rcv_real_all', 'sig_rcv_imag_all', 'bw_param_all', 'BW_actual_all', ...
    'rcv_pos_val', 'N_power_W_val', 'N_power_dBm_val', ...
    'exp_n_src_val', 'exp_bw_range_val', 'exp_n_per_bw_val', ...
    'exp_snr_dB_val', 'exp_distance_val', 'exp_azimuths_val', ...
    'symbolRate_src1_val', ...
    'N_sub_val', 'max_src_val', 'edge_val', 'lamda_val', ...
    'B_win_val', 'B_step_val', 'fs_val', 'symbolRate_val', ...
    'sub_f_lo_val', 'sub_f_hi_val', 'thresh_val', 'num_count_classes', ...
    '-v7.3');
fprintf('已保存: %s (%.1f GB)\n', save_file, dir(save_file).bytes/1e9);
fprintf('\n完成！%.1f 秒\n', toc);
