%% gen_exp2B.m  Exp 2-B: 两源频率间距扫描
%
% 实验目的:
%   展示DL、ED、特征值方法在两源从同频到分离时的源数估计能力
%
% 固定条件:
%   2源，位置 30°和200°，距中心1500m
%   每trial根据实际噪底和目标SNR反推Pt（保证实际SNR=目标值）
%
% SNR定义:
%   每源子带SNR = alpha_capture × Pr_avg / N_sub_noise
%   即该源独占子带中心时的信噪比
%   同频时两源在同一子带叠加，总子带SNR比单源高约3dB
%
% 扫描: 频率间距 Δf = [0, 0.2, 0.4, ..., 4.0] × symbolRate
% 每点: 500样本
% 输出: ctrl_exp2B.mat
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
%  Exp 2-B 实验参数
%% ═══════════════════════════════════════
n_per_point = 500;

% 固定信源位置（2源，等中心距离1500m）
d_center = 1500;
src_pos_2src = [d_center*cosd(30),  d_center*sind(30);    % 源1: 30°
                d_center*cosd(60),  d_center*sind(60)];   % 源2: 60°

% 固定实际每源子带SNR
target_snr_dB = 15;    % +15dB
snr_lin = 10^(target_snr_dB/10);

% 信号对齐子带中心时的捕获比例
alpha_capture = 0.955;

% 频率间距扫描（×symbolRate）
delta_f_ratio = 0 : 0.2 : 4.0;    % 21个点，等间距0.2×SR
n_delta = length(delta_f_ratio);

% 预计算各源到各站的路损（固定不变）
sta_dists = zeros(2, rcv_num);
PL_dB_all = zeros(2, rcv_num);
avg_Pr_per_Pt = zeros(1, 2);
for s = 1:2
    for m = 1:rcv_num
        sta_dists(s,m) = norm(src_pos_2src(s,:) - rcvPos(m,:));
        PL_dB_all(s,m) = PL_free(fc, sta_dists(s,m), 0, 0);
        avg_Pr_per_Pt(s) = avg_Pr_per_Pt(s) + 10^(-PL_dB_all(s,m)/10);
    end
    avg_Pr_per_Pt(s) = avg_Pr_per_Pt(s) / rcv_num;
end

%% ── 打印配置 ──
fprintf('═══════════════════════════════════════\n');
fprintf('  Exp 2-B: 两源频率间距扫描\n');
fprintf('═══════════════════════════════════════\n');
fprintf('每源目标子带SNR: %+d dB（每trial按实际噪底反推Pt）\n', target_snr_dB);
fprintf('噪底波动: 高斯 σ²=%.1f dB²\n', noise_jitter_var);
fprintf('中心距离: %.0fm\n', d_center);
fprintf('每间距点: %d样本  总计: %d样本\n\n', n_per_point, n_delta*n_per_point);

for s = 1:2
    fprintf('  源%d (%+.0f,%+.0f) %.0f°: 距离 [%.0f %.0f %.0f %.0f]m  avg_Pr/Pt=%.4e\n', ...
        s, src_pos_2src(s,1), src_pos_2src(s,2), ...
        atan2d(src_pos_2src(s,2), src_pos_2src(s,1)), ...
        sta_dists(s,1), sta_dists(s,2), sta_dists(s,3), sta_dists(s,4), ...
        avg_Pr_per_Pt(s));
end
fprintf('  源间距: %.0fm\n\n', norm(src_pos_2src(1,:) - src_pos_2src(2,:)));

fprintf('  Δf/SR: [%s]\n', num2str(delta_f_ratio, '%.1f '));
fprintf('  Δf(MHz): [%s]\n\n', num2str(delta_f_ratio * symbolRate / 1e6, '%.0f '));

%% ═══════════════════════════════════════
%  预分配
%% ═══════════════════════════════════════
N_total = n_delta * n_per_point;

mtr_sub_all      = zeros(N_total, N_sub, num_gx, num_gy, 'single');
src_count_all    = 2 * ones(N_total, 1, 'int32');    % 全是2源
band_mask_all    = zeros(N_total, max_src, N_sub, 'single');
ignore_mask_all  = zeros(N_total, max_src, N_sub, 'single');
avg_snr_all      = single(target_snr_dB) * ones(N_total, 1, 'single');
sub_energy_all   = zeros(N_total, N_sub, 'single');
cov_mat_real_all = zeros(N_total, N_sub, rcv_num, rcv_num, 'single');
cov_mat_imag_all = zeros(N_total, N_sub, rcv_num, rcv_num, 'single');
delta_f_all      = zeros(N_total, 1, 'single');       % 频率间距(×SR)
fc_offset_all    = zeros(N_total, max_src, 'single');  % 各源频偏
Pt_target_all    = zeros(N_total, max_src, 'single');  % 各源Pt
snr_target_all   = single(target_snr_dB) * ones(N_total, 1, 'single');

%% ═══════════════════════════════════════
%  生成数据
%% ═══════════════════════════════════════
idx = 0;
t_start = tic;

% 频偏范围（保证信号不出带）
fc_offset_limit = fs/2 - BW_actual/2;

for di = 1:n_delta
    delta_f = delta_f_ratio(di) * symbolRate;   % Hz
    
    % 中心频率的允许范围
    fc_center_min = -fc_offset_limit + delta_f/2;
    fc_center_max =  fc_offset_limit - delta_f/2;
    
    fprintf('  Δf=%.1f×SR (%.0fMHz)  fc_center∈[%+.1f, %+.1f]MHz ...', ...
        delta_f_ratio(di), delta_f/1e6, fc_center_min/1e6, fc_center_max/1e6);
    t0 = tic;
    
    for trial = 1:n_per_point
        idx = idx + 1;
        
        %% ── 随机中心频率，两源对称分布 ──
        fc_center = fc_center_min + (fc_center_max - fc_center_min) * rand();
        fc_off = [fc_center - delta_f/2, fc_center + delta_f/2];
        
        %% ── 本trial的标签（依赖fc_off）──
        band_mask_loc = zeros(max_src, N_sub, 'single');
        ignore_mask_loc = zeros(max_src, N_sub, 'single');
        for s = 1:2
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
        
        %% ── 根据实际噪底反推各源Pt（保证每源实际SNR=目标值）──
        Pt_each = zeros(1, 2);
        for s = 1:2
            Pt_each(s) = snr_lin * N_power_sub_trial / (alpha_capture * avg_Pr_per_Pt(s));
        end
        
        %% ── 各源独立基带信号 ──
        sig_pool = zeros(2, len);
        for s = 1:2
            [~, tmp] = Gen_basesig(dataLen, fs, Txobj.txId_V);
            tmp = tmp(1:len);
            sig_pool(s,:) = tmp / sqrt(mean(abs(tmp).^2));
        end
        
        %% ── 搬频 ──
        t_vec = (0:len-1) / fs;
        for s = 1:2
            sig_pool(s,:) = sig_pool(s,:) .* exp(1j*2*pi*fc_off(s)*t_vec);
            sig_pool(s,:) = sig_pool(s,:) / sqrt(mean(abs(sig_pool(s,:)).^2));
        end
        
        %% ── 构造接收信号（2源叠加 + 噪声）──
        sig_rcv = zeros(rcv_num, len);
        for m = 1:rcv_num
            noise = sqrt(N_power_W_trial/2) * (randn(1,len) + 1j*randn(1,len));
            sig_m = noise;
            for s = 1:2
                tau_m = sta_dists(s,m) / vc;
                sig_del = apply_delay_fd(sig_pool(s,:), tau_m, fs);
                Pr_W = Pt_each(s) * 10^(-PL_dB_all(s,m)/10);
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
        band_mask_all(idx,:,:)       = band_mask_loc;
        ignore_mask_all(idx,:,:)     = ignore_mask_loc;
        sub_energy_all(idx,:)        = sub_energy_loc;
        cov_mat_real_all(idx,:,:,:)  = cov_r;
        cov_mat_imag_all(idx,:,:,:)  = cov_i;
        delta_f_all(idx)             = single(delta_f_ratio(di));
        fc_offset_all(idx, 1:2)      = single(fc_off);
        Pt_target_all(idx, 1:2)      = single(Pt_each);
    end
    
    wait(gpuDevice); gpuDevice;
    fprintf(' %.1fs\n', toc(t0));
end

fprintf('\n生成完成！%d样本 | 总耗时 %.1f秒\n', N_total, toc(t_start));

%% ═══════════════════════════════════════
%  保存
%% ═══════════════════════════════════════
save('ctrl_exp2B.mat', ...
    'mtr_sub_all', 'src_count_all', 'band_mask_all', 'ignore_mask_all', ...
    'avg_snr_all', 'sub_energy_all', 'cov_mat_real_all', 'cov_mat_imag_all', ...
    'delta_f_all', 'fc_offset_all', 'Pt_target_all', 'snr_target_all', ...
    'N_sub_val', 'max_src_val', 'num_grid', ...
    'edge_val', 'lamda_val', 'B_win_val', 'B_step_val', ...
    'fs_val', 'symbolRate_val', 'sub_f_lo_val', 'sub_f_hi_val', ...
    'thresh_val', 'num_count_classes', ...
    '-v7.3');
fprintf('已保存 ctrl_exp2B.mat\n');

%% ── 打印统计 ──
fprintf('\n========== 各频率间距统计 ==========\n');
for di = 1:n_delta
    mask = (delta_f_all == delta_f_ratio(di));
    fc_vals = fc_offset_all(mask, :);
    fc_centers = mean(fc_vals, 2);
    n_bands = squeeze(sum(band_mask_all(mask, 1, :) + band_mask_all(mask, 2, :), 3));
    fprintf('Δf=%.1f×SR: fc_center∈[%+.1f, %+.1f]MHz  平均占用子带=%.1f\n', ...
        delta_f_ratio(di), min(fc_centers)/1e6, max(fc_centers)/1e6, mean(n_bands));
end