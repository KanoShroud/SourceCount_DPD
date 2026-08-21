%% gen_exp2C.m  Exp 2-C: 同频多源 SNR 扫描
%
% 实验目的:
%   展示DL在同频1/2/3源场景下的计数能力随SNR的变化
%
% 固定条件:
%   Δf=0 (全部同频), 随机中心频率
%   1源: 30°, 1500m
%   2源: 30°/60°, 1500m
%   3源: 30°/60°/90°, 1500m
%   每trial根据实际噪底和目标SNR反推Pt
%
% SNR定义:
%   每源子带SNR = alpha_capture × Pr_avg / N_sub_noise
%
% 扫描: 每源子带SNR -15 ~ +5 dB, 步长1dB, 21点
% 每点: 1000样本
% 输出: ctrl_exp2C.mat
%
clc; clear; close all;
tic
global Txobj Rxobj

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fileparts(script_dir));
runtime = gate0_runtime('chapter3', mfilename);

if isempty(gcp('nocreate'))
    parpool('local');
end
gpuDevice(1);

%% ═══════════════════════════════════════
%  基础参数（和main30一致）
%% ═══════════════════════════════════════
fc          = 5800e6;
arfa_V      = 0.25;
symbolRate  = 10e6;
BW_actual   = symbolRate * (1 + arfa_V * 1.2);
fs          = 100e6;
B_win       = 10e6;
B_step      = 5e6;
N_sub       = floor((fs - B_win) / B_step) + 1;   % = 19
len         = 2^12;
f_axis      = (-len/2 : len/2-1) * (fs/len);
max_src     = 3;
vc          = 299792458;
thresh      = 0.3;

N_power_dBm = -90;
N_power_dBW = N_power_dBm - 30;
N_power_W   = 10^(N_power_dBW / 10);
noise_jitter_var = 5;     % 噪底波动方差 (dB²)，高斯分布

init_pos = [0, 0];
edge     = 2000;
lamda    = 50;

%% ── 接收站 ──
R_rcv  = 500; N_rx = 4;
angles_rx = (0:N_rx-1) * 2*pi/N_rx;
rcvPos = [R_rcv*cos(angles_rx)', R_rcv*sin(angles_rx)'];
rcv_num = N_rx;

Rxobj.Num = 4;
Rxobj.rxId_V = ["001";"002";"003";"004"];
Rxobj.node_pos = [rcvPos, zeros(N_rx,1)];
Rxobj.sample_rate = repmat(fs, N_rx, 1);
Rxobj.freq_rf = repmat(fc, N_rx, 1);

%% ── Txobj模板 ──
Txobj.Num = 1; Txobj.txId_V = "drone001"; Txobj.freqC_V = fc;
Txobj.modType_V = "BPSK"; Txobj.multiplexingType_V = "NONE";
Txobj.shapingType_V = "RootRaisedCos"; Txobj.arfa_V = arfa_V;
Txobj.symbolRate_V = symbolRate; Txobj.modDepth_V = 1.0;
Txobj.contPhase_V = "cont"; Txobj.nodePos_V = [0,0,0]; Txobj.Gt = 0;
Txobj.antennaType_V = "AntennaX"; Txobj.antennaDeg_V = [0,0];
Txobj.transmitPower_V = 1.0;
Txobj.txTime_V = "2020-06-06 00:00:00:000000";
Txobj.txDuration_V = 0; Txobj.Bw_V = 0;

%% ── 子带掩模 ──
sub_masks = false(N_sub, len);
sub_f_lo = zeros(1, N_sub); sub_f_hi = zeros(1, N_sub);
for k = 1:N_sub
    sub_f_lo(k) = (k-1)*B_step - fs/2;
    sub_f_hi(k) = (k-1)*B_step - fs/2 + B_win;
    sub_masks(k,:) = (f_axis >= sub_f_lo(k)) & (f_axis < sub_f_hi(k));
end

num_gx = length(init_pos(1)-edge:lamda:init_pos(1)+edge);
num_gy = num_gx;
dataLen = calc_dataLen(len, fs, Txobj.symbolRate_V);

%% ── 配置参数（Python可读）──
N_sub_val = int32(N_sub); max_src_val = int32(max_src);
num_grid = int32(num_gx);
edge_val = single(edge); lamda_val = single(lamda);
B_win_val = single(B_win); B_step_val = single(B_step);
fs_val = single(fs); symbolRate_val = single(symbolRate);
sub_f_lo_val = single(sub_f_lo); sub_f_hi_val = single(sub_f_hi);
thresh_val = single(thresh);
num_count_classes = int32(max_src + 1);

%% ═══════════════════════════════════════
%  Exp 2-C 实验参数
%% ═══════════════════════════════════════
if runtime.is_smoke
    n_per_point = 1;
else
    n_per_point = 500;
end

% 频偏范围（保证信号不出带）
fc_offset_limit = fs/2 - BW_actual/2;

% 信号对齐子带中心时的捕获比例
alpha_capture = 0.955;

% 目标每源子带SNR
target_snr_dB = (-15 : 1 : 5)';
n_snr = length(target_snr_dB);

% 源数场景: 1源、2源同频、3源同频
d_center = 1500;
src_configs = {
    [d_center*cosd(30), d_center*sind(30)];                                % 1源: 30°
    [d_center*cosd(30), d_center*sind(30);                                 % 2源: 30°/60°
     d_center*cosd(60), d_center*sind(60)];
    [d_center*cosd(30), d_center*sind(30);                                 % 3源: 30°/60°/90°
     d_center*cosd(60), d_center*sind(60);
     d_center*cosd(90), d_center*sind(90)];
};
n_configs = length(src_configs);

%% ── 预计算各配置的路损 ──
config_info = cell(n_configs, 1);
for ci = 1:n_configs
    src_pos = src_configs{ci};
    n_src = size(src_pos, 1);
    sta_dists = zeros(n_src, rcv_num);
    PL_dB = zeros(n_src, rcv_num);
    avg_Pr_per_Pt = zeros(1, n_src);
    for s = 1:n_src
        for m = 1:rcv_num
            sta_dists(s,m) = norm(src_pos(s,:) - rcvPos(m,:));
            PL_dB(s,m) = PL_free(fc, sta_dists(s,m), 0, 0);
            avg_Pr_per_Pt(s) = avg_Pr_per_Pt(s) + 10^(-PL_dB(s,m)/10);
        end
        avg_Pr_per_Pt(s) = avg_Pr_per_Pt(s) / rcv_num;
    end
    config_info{ci}.n_src = n_src;
    config_info{ci}.src_pos = src_pos;
    config_info{ci}.sta_dists = sta_dists;
    config_info{ci}.PL_dB = PL_dB;
    config_info{ci}.avg_Pr_per_Pt = avg_Pr_per_Pt;
end

%% ── 打印配置 ──
fprintf('═══════════════════════════════════════\n');
fprintf('  Exp 2-C: 同频多源 SNR 扫描\n');
fprintf('═══════════════════════════════════════\n');
fprintf('Δf=0 (全部同频), 随机中心频率\n');
fprintf('噪底波动: 高斯 σ²=%.1f dB²\n', noise_jitter_var);
fprintf('每SNR点: %d样本\n', n_per_point);
fprintf('SNR范围: %+d ~ %+d dB, %d点\n\n', target_snr_dB(1), target_snr_dB(end), n_snr);

for ci = 1:n_configs
    info = config_info{ci};
    fprintf('  配置%d (%d源):\n', ci, info.n_src);
    for s = 1:info.n_src
        fprintf('    源%d (%+.0f,%+.0f) %.0f°  avg_Pr/Pt=%.4e\n', ...
            s, info.src_pos(s,1), info.src_pos(s,2), ...
            atan2d(info.src_pos(s,2), info.src_pos(s,1)), ...
            info.avg_Pr_per_Pt(s));
    end
end
fprintf('\n  总样本数: %d\n\n', n_configs * n_snr * n_per_point);

%% ═══════════════════════════════════════
%  预分配
%% ═══════════════════════════════════════
N_total = n_configs * n_snr * n_per_point;

mtr_sub_all      = zeros(N_total, N_sub, num_gx, num_gy, 'single');
src_count_all    = zeros(N_total, 1, 'int32');
band_mask_all    = zeros(N_total, max_src, N_sub, 'single');
ignore_mask_all  = zeros(N_total, max_src, N_sub, 'single');
avg_snr_all      = zeros(N_total, 1, 'single');
sub_energy_all   = zeros(N_total, N_sub, 'single');
cov_mat_real_all = zeros(N_total, N_sub, rcv_num, rcv_num, 'single');
cov_mat_imag_all = zeros(N_total, N_sub, rcv_num, rcv_num, 'single');
fc_offset_all    = zeros(N_total, max_src, 'single');
Pt_target_all    = zeros(N_total, max_src, 'single');
snr_target_all   = zeros(N_total, 1, 'single');
config_id_all    = zeros(N_total, 1, 'int32');     % 1/2/3 = 源数

%% ═══════════════════════════════════════
%  生成数据
%% ═══════════════════════════════════════
idx = 0;
t_start = tic;

for ci = 1:n_configs
    info = config_info{ci};
    n_src = info.n_src;
    fprintf('===== 配置%d: %d源同频 =====\n', ci, n_src);
    
    for si = 1:n_snr
        snr_lin = 10^(target_snr_dB(si)/10);
        fprintf('  SNR=%+ddB ...', target_snr_dB(si));
        t0 = tic;
        
        for trial = 1:n_per_point
            idx = idx + 1;
            
            %% ── 随机中心频率（Δf=0, 所有源共享同一频偏）──
            fc_center = -fc_offset_limit + 2*fc_offset_limit * rand();
            fc_off = repmat(fc_center, 1, n_src);   % 同频
            
            %% ── 本trial的标签 ──
            band_mask_loc = zeros(max_src, N_sub, 'single');
            ignore_mask_loc = zeros(max_src, N_sub, 'single');
            for s = 1:n_src
                mainlobe_lo = fc_off(s) - symbolRate/2;
                mainlobe_hi = fc_off(s) + symbolRate/2;
                rolloff_lo  = fc_off(s) - BW_actual/2;
                rolloff_hi  = fc_off(s) + BW_actual/2;
                for k = 1:N_sub
                    ov_main = max(0, min(mainlobe_hi, sub_f_hi(k)) ...
                                  - max(mainlobe_lo, sub_f_lo(k)));
                    ov_all  = max(0, min(rolloff_hi, sub_f_hi(k)) ...
                                  - max(rolloff_lo, sub_f_lo(k)));
                    ov_roll = ov_all - ov_main;
                    cov_ratio = ov_main / B_win;
                    if cov_ratio >= thresh
                        band_mask_loc(s, k) = 1;
                    elseif cov_ratio > 0 || ov_roll > 0
                        ignore_mask_loc(s, k) = 1;
                    end
                end
            end
            
            %% ── 本trial的噪底（4站共同波动）──
            N_power_W_trial = N_power_W * 10^(sqrt(noise_jitter_var) * randn() / 10);
            N_power_sub_trial = N_power_W_trial * (B_win / fs);
            
            %% ── 根据实际噪底反推各源Pt ──
            Pt_each = zeros(1, n_src);
            for s = 1:n_src
                Pt_each(s) = snr_lin * N_power_sub_trial / (alpha_capture * info.avg_Pr_per_Pt(s));
            end
            
            %% ── 各源独立基带信号 ──
            sig_pool = zeros(n_src, len);
            for s = 1:n_src
                [~, tmp] = Gen_basesig(dataLen, fs, Txobj.txId_V);
                tmp = tmp(1:len);
                sig_pool(s,:) = tmp / sqrt(mean(abs(tmp).^2));
            end
            
            %% ── 搬频（同频，共享fc_center）──
            t_vec = (0:len-1) / fs;
            for s = 1:n_src
                sig_pool(s,:) = sig_pool(s,:) .* exp(1j*2*pi*fc_off(s)*t_vec);
                sig_pool(s,:) = sig_pool(s,:) / sqrt(mean(abs(sig_pool(s,:)).^2));
            end
            
            %% ── 构造接收信号 ──
            sig_rcv = zeros(rcv_num, len);
            for m = 1:rcv_num
                noise = sqrt(N_power_W_trial/2) * (randn(1,len) + 1j*randn(1,len));
                sig_m = noise;
                for s = 1:n_src
                    tau_m = info.sta_dists(s,m) / vc;
                    sig_del = apply_delay_fd(sig_pool(s,:), tau_m, fs);
                    Pr_W = Pt_each(s) * 10^(-info.PL_dB(s,m)/10);
                    sig_m = sig_m + sqrt(Pr_W) * sig_del;
                end
                sig_rcv(m,:) = sig_m;
            end
            
            %% ── FFT + 子带 ──
            data_fft = zeros(rcv_num, len);
            for m = 1:rcv_num
                data_fft(m,:) = fftshift(fft(sig_rcv(m,:)));
            end
            
            data_fft_batch = zeros(N_sub, rcv_num, len);
            sub_energy_loc = zeros(1, N_sub, 'single');
            cov_r = zeros(N_sub, rcv_num, rcv_num, 'single');
            cov_i = zeros(N_sub, rcv_num, rcv_num, 'single');
            
            for k = 1:N_sub
                sig_sub = zeros(rcv_num, len);
                Pk_sum = 0;
                for m = 1:rcv_num
                    fft_k = data_fft(m,:) .* sub_masks(k,:);
                    sig_k = ifft(ifftshift(fft_k));
                    sig_sub(m,:) = sig_k;
                    P_k = mean(abs(sig_k).^2);
                    Pk_sum = Pk_sum + P_k;
                    if P_k > 0, sig_k = sig_k / sqrt(P_k); end
                    data_fft_batch(k,m,:) = fftshift(fft(sig_k));
                end
                sub_energy_loc(k) = Pk_sum / rcv_num;
                R_k = (sig_sub * sig_sub');
                cov_r(k,:,:) = single(real(R_k));
                cov_i(k,:,:) = single(imag(R_k));
            end
            
            %% ── DPD ──
            mtr_sub_loc = single(DPD_calculator_gpu_batch(...
                rcvPos, data_fft_batch, init_pos, edge, lamda, fs));
            
            %% ── 写入 ──
            mtr_sub_all(idx,:,:,:)       = mtr_sub_loc;
            src_count_all(idx)           = int32(n_src);
            band_mask_all(idx,:,:)       = band_mask_loc;
            ignore_mask_all(idx,:,:)     = ignore_mask_loc;
            avg_snr_all(idx)             = single(target_snr_dB(si));
            sub_energy_all(idx,:)        = sub_energy_loc;
            cov_mat_real_all(idx,:,:,:)  = cov_r;
            cov_mat_imag_all(idx,:,:,:)  = cov_i;
            fc_offset_all(idx, 1:n_src)  = single(fc_off);
            Pt_target_all(idx, 1:n_src)  = single(Pt_each);
            snr_target_all(idx)          = single(target_snr_dB(si));
            config_id_all(idx)           = int32(n_src);
        end
        
        wait(gpuDevice); gpuDevice;
        fprintf(' %.1fs\n', toc(t0));
    end
end

fprintf('\n生成完成！%d样本 | 总耗时 %.1f秒\n', N_total, toc(t_start));

%% ═══════════════════════════════════════
%  保存
%% ═══════════════════════════════════════
save(fullfile(runtime.data_dir, 'ctrl_exp2C.mat'), ...
    'mtr_sub_all', 'src_count_all', 'band_mask_all', 'ignore_mask_all', ...
    'avg_snr_all', 'sub_energy_all', 'cov_mat_real_all', 'cov_mat_imag_all', ...
    'fc_offset_all', 'Pt_target_all', 'snr_target_all', 'config_id_all', ...
    'N_sub_val', 'max_src_val', 'num_grid', ...
    'edge_val', 'lamda_val', 'B_win_val', 'B_step_val', ...
    'fs_val', 'symbolRate_val', 'sub_f_lo_val', 'sub_f_hi_val', ...
    'thresh_val', 'num_count_classes', ...
    '-v7.3');
fprintf('已保存 ctrl_exp2C.mat\n');
