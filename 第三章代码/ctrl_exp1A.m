%% gen_exp1A.m  Exp 1-A: 单源检测 Pd vs 子带SNR
%
% 固定条件:
%   信源位置 (1600, 700)
%   信号对齐子带W10中心 (fc_offset = 0 MHz)
%   每trial根据实际噪底和目标SNR反推Pt
%
% 扫描: 子带SNR -15 ~ +5 dB, 步长1dB, 21点
% 每点: 500样本 (随机: BPSK序列 + 噪声)
% 输出: ctrl_exp1A.mat
%
clc; clear; close all;
tic
global Txobj Rxobj

if isempty(gcp('nocreate'))
    parpool('local');
end
gpuDevice(1);

%% ═══════════════════════════════════════
%  基础参数（和main27一致）
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
N_power_W   = 10^(N_power_dBW / 10);       % 1e-12
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
%  Exp 1-A 实验参数
%% ═══════════════════════════════════════
n_per_point = 1000;

% 固定信源位置
src_pos_fixed = [1600, 700];

% 固定频偏: 对齐子带W10中心
fc_offset_fixed = 0;

% 信号对齐子带中心时的捕获比例
alpha_capture = 0.955;

% 目标子带SNR
target_snr_dB = (-15 : 1 : 15)';
n_snr = length(target_snr_dB);

% 预计算各站路损（固定不变）
sta_dists = zeros(1, rcv_num);
PL_dB_sta = zeros(1, rcv_num);
avg_Pr_per_Pt = 0;
for m = 1:rcv_num
    sta_dists(m) = norm(src_pos_fixed - rcvPos(m,:));
    PL_dB_sta(m) = PL_free(fc, sta_dists(m), 0, 0);
    avg_Pr_per_Pt = avg_Pr_per_Pt + 10^(-PL_dB_sta(m)/10);
end
avg_Pr_per_Pt = avg_Pr_per_Pt / rcv_num;

% 标签预计算（位置和频偏固定，所有样本共享）
band_mask_fixed = zeros(max_src, N_sub, 'single');
ignore_mask_fixed = zeros(max_src, N_sub, 'single');
mainlobe_lo = fc_offset_fixed - symbolRate/2;
mainlobe_hi = fc_offset_fixed + symbolRate/2;
rolloff_lo  = fc_offset_fixed - BW_actual/2;
rolloff_hi  = fc_offset_fixed + BW_actual/2;
for k = 1:N_sub
    ov_main = max(0, min(mainlobe_hi, sub_f_hi(k)) ...
                  - max(mainlobe_lo, sub_f_lo(k)));
    ov_all  = max(0, min(rolloff_hi, sub_f_hi(k)) ...
                  - max(rolloff_lo, sub_f_lo(k)));
    cov_ratio = ov_main / B_win;
    if cov_ratio >= thresh
        band_mask_fixed(1,k) = 1;
    elseif cov_ratio > 0 || (ov_all - ov_main) > 0
        ignore_mask_fixed(1,k) = 1;
    end
end

%% ── 打印配置 ──
fprintf('═══════════════════════════════════════\n');
fprintf('  Exp 1-A: 单源 Pd vs 子带SNR\n');
fprintf('═══════════════════════════════════════\n');
fprintf('信源: (%.2f, %.2f)m  距中心%.0fm  %.1f度\n', ...
    src_pos_fixed(1), src_pos_fixed(2), norm(src_pos_fixed), ...
    atan2d(src_pos_fixed(2), src_pos_fixed(1)));
fprintf('频偏: %.0fMHz (对齐W10)  alpha=%.3f\n', ...
    fc_offset_fixed/1e6, alpha_capture);
fprintf('噪底波动: 高斯 σ²=%.1f dB²（每trial按实际噪底反推Pt）\n', noise_jitter_var);
fprintf('avg_Pr_per_Pt: %.4e\n', avg_Pr_per_Pt);
fprintf('每点: %d样本  总计: %d样本\n\n', n_per_point, n_snr*n_per_point);

for m = 1:rcv_num
    fprintf('  站%d (%+.0f,%+.0f): dist=%.1fm  PL=%.1fdB\n', ...
        m, rcvPos(m,1), rcvPos(m,2), sta_dists(m), PL_dB_sta(m));
end
fprintf('\nband_mask:  [%s]\n', num2str(band_mask_fixed(1,:), '%d '));
fprintf('ignore:     [%s]\n\n', num2str(ignore_mask_fixed(1,:), '%d '));

%% ═══════════════════════════════════════
%  预分配
%% ═══════════════════════════════════════
N_total = n_snr * n_per_point;

mtr_sub_all      = zeros(N_total, N_sub, num_gx, num_gy, 'single');
src_count_all    = ones(N_total, 1, 'int32');
band_mask_all    = zeros(N_total, max_src, N_sub, 'single');
ignore_mask_all  = zeros(N_total, max_src, N_sub, 'single');
avg_snr_all      = zeros(N_total, 1, 'single');
sub_energy_all   = zeros(N_total, N_sub, 'single');
cov_mat_real_all = zeros(N_total, N_sub, rcv_num, rcv_num, 'single');
cov_mat_imag_all = zeros(N_total, N_sub, rcv_num, rcv_num, 'single');
Pt_target_all    = zeros(N_total, 1, 'single');
snr_target_all   = zeros(N_total, 1, 'single');

%% ═══════════════════════════════════════
%  生成数据
%% ═══════════════════════════════════════
idx = 0;
t_start = tic;

for si = 1:n_snr
    snr_lin = 10^(target_snr_dB(si)/10);
    fprintf('  SNR=%+ddB ...', target_snr_dB(si));
    t0 = tic;

    for trial = 1:n_per_point
        idx = idx + 1;

        %% ── 本trial的噪底（4站共同波动）──
        N_power_W_trial = N_power_W * 10^(sqrt(noise_jitter_var) * randn() / 10);
        N_power_sub_trial = N_power_W_trial * (B_win / fs);

        %% ── 根据实际噪底反推Pt（保证实际SNR=目标值）──
        Pt = snr_lin * N_power_sub_trial / (alpha_capture * avg_Pr_per_Pt);

        %% ── 各站接收功率（本trial的Pt）──
        Pr_W_sta = zeros(1, rcv_num);
        for m = 1:rcv_num
            Pr_W_sta(m) = Pt * 10^(-PL_dB_sta(m)/10);
        end

        %% ── 基带信号 ──
        [~, tmp] = Gen_basesig(dataLen, fs, Txobj.txId_V);
        tmp = tmp(1:len);
        baseSig = tmp / sqrt(mean(abs(tmp).^2));

        %% ── 搬频 ──
        t_vec = (0:len-1) / fs;
        baseSig = baseSig .* exp(1j * 2*pi * fc_offset_fixed * t_vec);
        baseSig = baseSig / sqrt(mean(abs(baseSig).^2));

        %% ── 接收信号 ──
        sig_rcv = zeros(rcv_num, len);
        for m = 1:rcv_num
            tau_m = sta_dists(m) / vc;
            sig_del = apply_delay_fd(baseSig, tau_m, fs);
            noise = sqrt(N_power_W_trial/2) * (randn(1,len) + 1j*randn(1,len));
            sig_rcv(m,:) = sqrt(Pr_W_sta(m)) * sig_del + noise;
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
        band_mask_all(idx,:,:)       = band_mask_fixed;
        ignore_mask_all(idx,:,:)     = ignore_mask_fixed;
        avg_snr_all(idx)             = single(target_snr_dB(si));
        sub_energy_all(idx,:)        = sub_energy_loc;
        cov_mat_real_all(idx,:,:,:)  = cov_r;
        cov_mat_imag_all(idx,:,:,:)  = cov_i;
        Pt_target_all(idx)           = single(Pt);
        snr_target_all(idx)          = single(target_snr_dB(si));
    end

    wait(gpuDevice); gpuDevice;
    fprintf(' %.1fs\n', toc(t0));
end

fprintf('\n生成完成！%d样本 | 总耗时 %.1f秒\n', N_total, toc(t_start));

%% ═══════════════════════════════════════
%  保存
%% ═══════════════════════════════════════
save('ctrl_exp1A.mat', ...
    'mtr_sub_all', 'src_count_all', 'band_mask_all', 'ignore_mask_all', ...
    'avg_snr_all', 'sub_energy_all', 'cov_mat_real_all', 'cov_mat_imag_all', ...
    'Pt_target_all', 'snr_target_all', ...
    'N_sub_val', 'max_src_val', 'num_grid', ...
    'edge_val', 'lamda_val', 'B_win_val', 'B_step_val', ...
    'fs_val', 'symbolRate_val', 'sub_f_lo_val', 'sub_f_hi_val', ...
    'thresh_val', 'num_count_classes', ...
    '-v7.3');
fprintf('已保存 ctrl_exp1A.mat\n');