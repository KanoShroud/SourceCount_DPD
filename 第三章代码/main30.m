%% main30.m  多源信源检测 + 实例级频段估计（直接输出Python可读格式）
%
% 输出文件：{set_name}_data.mat，包含：
%   mtr_sub_all:     (N × N_sub × gx × gy)    子带DPD空间谱
%   src_count_all:   (N × 1)                   信源总数
%   band_mask_all:   (N × max_src × N_sub)     子带占用标签
%   ignore_mask_all: (N × max_src × N_sub)     模糊子带
%   avg_snr_all:     (N × 1)                   最弱源平均SNR
%   fc_offset_all:   (N × max_src)             各源频偏
%   Pt_W_all:        (N × max_src)             各源功率
%   src_pos_all:     (N × max_src × 2)         各源位置
%   sub_energy_all:  (N × N_sub)               子带能量（归一化前）
%   cov_mat_real_all:(N × N_sub × M × M)       子带协方差矩阵实部（总能量，未除len）
%   cov_mat_imag_all:(N × N_sub × M × M)       子带协方差矩阵虚部（总能量，未除len）
%   + 配置参数
%
clc; clear; close all;
tic
global Txobj Rxobj

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fileparts(script_dir));
runtime = gate0_runtime('chapter3', mfilename);

if runtime.is_smoke
    fprintf('[Gate1] smoke 模式跳过未使用的 MATLAB 并行池。\n');
elseif isempty(gcp('nocreate'))
    parpool('local');
end
gpuDevice(1);

%% ═══════════════════════════════════════
%  参数配置区
%% ═══════════════════════════════════════
set_list = {'train', 'val', 'test'};
if runtime.is_smoke
    trials_list = [8, 4, 4];
    smoke_seed_val = int32(20260820);
    rng(double(smoke_seed_val), 'twister');
else
    trials_list = [40000, 5000, 5000];
    smoke_seed_val = int32(-1);
end

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

%% ── 信源场景配置 ──
max_src         = 3;
src_num_range   = [0, 1, 2, 3];
src_num_weights = [0.20, 0.20, 0.30, 0.30];

fc_offset_min = -(fs/2 - BW_actual/2);
fc_offset_max =  (fs/2 - BW_actual/2);

%% ── 功率配置（W单位）──
Pt_W_range = [0.003, 0.05];
Pt_jitter_dB = 2;

N_power_dBm = -90;
N_power_dBW = N_power_dBm - 30;
N_power_W   = 10^(N_power_dBW / 10);
noise_jitter_var = 5;     % 噪底波动方差 (dB²)，高斯分布

%% ── 空间配置 ──
dist_range         = [0, 2000];
dist_jitter_ratio  = 0.1;
min_dist_src2src   = 300;

init_pos = [0, 0];
edge     = 2000;
lamda    = 50;
vc       = 299792458;

%% ── 标签阈值 ──
thresh = 0.3;

%% ═══════════════════════════════════════
%  构造 Rxobj
%% ═══════════════════════════════════════
Rxobj.Num         = 4;
Rxobj.rxId_V      = ["001";"002";"003";"004"];
R_rcv  = 500;
N_rx   = 4;
angles_rx = (0:N_rx-1) * 2*pi/N_rx;
Rxobj.node_pos    = [R_rcv*cos(angles_rx)', R_rcv*sin(angles_rx)', zeros(N_rx,1)];
Rxobj.sample_rate = repmat(fs, Rxobj.Num, 1);
Rxobj.freq_rf     = repmat(fc, Rxobj.Num, 1);

rcvPos  = Rxobj.node_pos(:, 1:2);
rcv_num = Rxobj.Num;

%% ═══════════════════════════════════════
%  构造 Txobj
%% ═══════════════════════════════════════
Txobj.Num                = 1;
Txobj.txId_V             = "drone001";
Txobj.freqC_V            = fc;
Txobj.modType_V          = "BPSK";
Txobj.multiplexingType_V = "NONE";
Txobj.shapingType_V      = "RootRaisedCos";
Txobj.arfa_V             = arfa_V;
Txobj.symbolRate_V       = symbolRate;
Txobj.modDepth_V         = 1.0;
Txobj.contPhase_V        = "cont";
Txobj.nodePos_V          = [0, 0, 0];
Txobj.Gt                 = 0;
Txobj.antennaType_V      = "AntennaX";
Txobj.antennaDeg_V       = [0, 0];
Txobj.transmitPower_V    = 1.0;
Txobj.txTime_V           = "2020-06-06 00:00:00:000000";
Txobj.txDuration_V       = 0;
Txobj.Bw_V               = 0;

%% ═══════════════════════════════════════
%  子带掩模
%% ═══════════════════════════════════════
sub_masks = false(N_sub, len);
sub_f_lo  = zeros(1, N_sub);
sub_f_hi  = zeros(1, N_sub);
for k = 1:N_sub
    sub_f_lo(k) = (k-1)*B_step - fs/2;
    sub_f_hi(k) = (k-1)*B_step - fs/2 + B_win;
    sub_masks(k,:) = (f_axis >= sub_f_lo(k)) & (f_axis < sub_f_hi(k));
end

%% ── 网格尺寸 ──
num_gx = length(init_pos(1)-edge : lamda : init_pos(1)+edge);
num_gy = num_gx;

fprintf('═══════════════════════════════════════\n');
fprintf('  main30  配置\n');
fprintf('═══════════════════════════════════════\n');
fprintf('采样率: %.0fMHz  子带: B_win=%.0fMHz  N_sub=%d\n', fs/1e6, B_win/1e6, N_sub);
fprintf('搜索区域: ±%.0fm  网格步长: %.0fm  网格: %d×%d\n', ...
        edge, lamda, num_gx, num_gy);
fprintf('接收站: %d个，R=%.0fm 均匀圆\n', rcv_num, R_rcv);
fprintf('信源距离: [%.0f, %.0f]m ±%.0f%%扰动  功率: [%.3f, %.3f]W ±%.0fdB扰动\n', ...
        dist_range(1), dist_range(2), dist_jitter_ratio*100, ...
        Pt_W_range(1), Pt_W_range(2), Pt_jitter_dB);
fprintf('噪底波动: 高斯 σ²=%.1f dB²\n', noise_jitter_var);
fprintf('标签阈值: thresh=%.2f\n', thresh);
fprintf('max_src=%d  输出格式: 直接Python可读\n', max_src);
fprintf('\n');

dataLen = calc_dataLen(len, fs, Txobj.symbolRate_V);

%% ═══════════════════════════════════════
%  保存为Python可读的配置参数
%% ═══════════════════════════════════════
N_sub_val         = int32(N_sub);
max_src_val       = int32(max_src);
num_grid          = int32(num_gx);
edge_val          = single(edge);
lamda_val         = single(lamda);
B_win_val         = single(B_win);
B_step_val        = single(B_step);
fs_val            = single(fs);
symbolRate_val    = single(symbolRate);
BW_actual_val     = single(BW_actual);
arfa_val          = single(arfa_V);
sub_f_lo_val      = single(sub_f_lo);
sub_f_hi_val      = single(sub_f_hi);
thresh_val        = single(thresh);
num_count_classes = int32(max_src + 1);

%% ═══════════════════════════════════════
%  三轮循环：train → val → test
%% ═══════════════════════════════════════
for si = 1:length(set_list)

    set_name = set_list{si};
    N_trials = trials_list(si);

    fprintf('\n========== 开始生成 %s 集 (%d 条) ==========\n', set_name, N_trials);
    if N_trials == 0, continue; end

    %% ── 信源数量按权重分配 ──
    src_num_seq = [];
    for c = 1:length(src_num_range)
        n_c = round(N_trials * src_num_weights(c));
        src_num_seq = [src_num_seq, repmat(src_num_range(c), 1, n_c)];
    end
    remainder = N_trials - length(src_num_seq);
    if remainder > 0
        src_num_seq = [src_num_seq, repmat(src_num_range(end), 1, remainder)];
    elseif remainder < 0
        src_num_seq = src_num_seq(1:N_trials);
    end
    src_num_seq = src_num_seq(randperm(N_trials));

    %% ── 预分配大矩阵 ──
    mtr_sub_all     = zeros(N_trials, N_sub, num_gx, num_gy, 'single');
    src_count_all   = zeros(N_trials, 1, 'int32');
    band_mask_all   = zeros(N_trials, max_src, N_sub, 'single');
    ignore_mask_all = zeros(N_trials, max_src, N_sub, 'single');
    avg_snr_all     = zeros(N_trials, 1, 'single');
    fc_offset_all   = zeros(N_trials, max_src, 'single');
    Pt_W_all        = zeros(N_trials, max_src, 'single');
    src_pos_all     = zeros(N_trials, max_src, 2, 'single');
    sub_energy_all  = zeros(N_trials, N_sub, 'single');
    cov_mat_real_all = zeros(N_trials, N_sub, rcv_num, rcv_num, 'single');
    cov_mat_imag_all = zeros(N_trials, N_sub, rcv_num, rcv_num, 'single');

    %% ── 统计计数器 ──
    stat_src_count = zeros(1, max_src + 1);
    stat_band1  = 0;
    stat_band0  = 0;
    stat_ignore = 0;

    t_start = tic;

    for trial = 1:N_trials

        n_src = src_num_seq(trial);

        %% ── 本trial的噪底（4站共同波动）──
        N_power_W_trial = N_power_W * 10^(sqrt(noise_jitter_var) * randn() / 10);

        if n_src == 0
            %% ════════════════════════════════
            %  0源：纯噪声
            %% ════════════════════════════════
            sig_rcv = zeros(rcv_num, len);
            for m = 1:rcv_num
                noise = sqrt(N_power_W_trial/2) * (randn(1,len) + 1j*randn(1,len));
                sig_rcv(m,:) = noise;
            end

            data_fft = zeros(rcv_num, len);
            for m = 1:rcv_num
                data_fft(m,:) = fftshift(fft(sig_rcv(m,:)));
            end

            data_fft_batch = zeros(N_sub, rcv_num, len);
            sub_energy_loc = zeros(1, N_sub, 'single');
            for k = 1:N_sub
                sig_sub = zeros(rcv_num, len);
                Pk_sum = 0;
                for m = 1:rcv_num
                    fft_k = data_fft(m,:) .* sub_masks(k,:);
                    sig_k = ifft(ifftshift(fft_k));
                    sig_sub(m,:) = sig_k;
                    P_k   = mean(abs(sig_k).^2);
                    Pk_sum = Pk_sum + P_k;
                    if P_k > 0
                        sig_k = sig_k / sqrt(P_k);
                    end
                    data_fft_batch(k, m, :) = fftshift(fft(sig_k));
                end
                sub_energy_loc(k) = Pk_sum / rcv_num;
                R_k = (sig_sub * sig_sub');
                cov_mat_real_all(trial, k, :, :) = single(real(R_k));
                cov_mat_imag_all(trial, k, :, :) = single(imag(R_k));
            end

            mtr_sub_loc = single(DPD_calculator_gpu_batch(...
                rcvPos, data_fft_batch, init_pos, edge, lamda, fs));

            mtr_sub_all(trial,:,:,:)     = mtr_sub_loc;
            src_count_all(trial)         = int32(0);
            sub_energy_all(trial,:)      = sub_energy_loc;
            avg_snr_all(trial)           = single(-999);

            stat_src_count(1) = stat_src_count(1) + 1;
            stat_band0 = stat_band0 + max_src * N_sub;

        else
            %% ════════════════════════════════
            %  有源
            %% ════════════════════════════════

            %% ── 生成基带信号 ──
            sig_pool = zeros(n_src, len);
            for s = 1:n_src
                [~, tmp] = Gen_basesig(dataLen, fs, Txobj.txId_V);
                tmp = tmp(1:len);
                sig_pool(s, :) = tmp / sqrt(mean(abs(tmp).^2));
            end

            %% ── 随机信源位置 ──
            dist_min_base = max(dist_range(1), n_src * min_dist_src2src / (2*pi) * 1.5);
            dist_base = dist_min_base + (dist_range(2) - dist_min_base) * rand();
            dist_lo = max(dist_range(1), dist_base * (1 - dist_jitter_ratio));
            dist_hi = min(dist_range(2), dist_base * (1 + dist_jitter_ratio));
            src_pos = gen_multi_source_pos_v2(...
                rcvPos, n_src, [dist_lo, dist_hi], min_dist_src2src);

            %% ── 随机中心频偏 ──
            fc_off = assign_freq_offsets(n_src, fc_offset_min, fc_offset_max, symbolRate);

            %% ── 随机发射功率 ──
            Pt_base_dBW = log10(Pt_W_range(1)) + ...
                         (log10(Pt_W_range(2))-log10(Pt_W_range(1))) * rand();
            Pt_each_dBW = Pt_base_dBW + Pt_jitter_dB/10 * (2*rand(1,n_src)-1);
            Pt_W   = 10.^(Pt_each_dBW);
            Pt_W   = max(Pt_W_range(1), min(Pt_W_range(2), Pt_W));
            Pt_dBW = 10*log10(Pt_W);

            %% ── 构造接收信号 ──
            sig_rcv_accum = zeros(rcv_num, len);
            snr_mat       = zeros(n_src, rcv_num);
            t_vec         = (0:len-1) / fs;

            for s = 1:n_src
                baseSig = sig_pool(s, :);
                baseSig = baseSig .* exp(1j * 2*pi * fc_off(s) * t_vec);
                baseSig = baseSig / sqrt(mean(abs(baseSig).^2));

                PL_dB = zeros(1, rcv_num);
                for m = 1:rcv_num
                    PL_dB(m) = PL_free(fc, ...
                        norm(src_pos(s,:) - rcvPos(m,:)), 0, 0);
                end

                for m = 1:rcv_num
                    tau_m   = norm(src_pos(s,:) - rcvPos(m,:)) / vc;
                    sig_del = apply_delay_fd(baseSig, tau_m, fs);
                    Pr_W    = Pt_W(s) * 10^(-PL_dB(m)/10);
                    sig_rcv_accum(m,:) = sig_rcv_accum(m,:) + sqrt(Pr_W) * sig_del;
                    snr_mat(s,m) = 10*log10(Pr_W / N_power_W);
                end
            end

            %% ── 加噪声 ──
            sig_rcv = zeros(rcv_num, len);
            for m = 1:rcv_num
                noise = sqrt(N_power_W_trial/2) * (randn(1,len) + 1j*randn(1,len));
                sig_rcv(m,:) = sig_rcv_accum(m,:) + noise;
            end

            %% ── 全带宽FFT ──
            data_fft = zeros(rcv_num, len);
            for m = 1:rcv_num
                data_fft(m,:) = fftshift(fft(sig_rcv(m,:)));
            end

            %% ── 子带滤波 + 子带归一化 ──
            data_fft_batch = zeros(N_sub, rcv_num, len);
            sub_energy_loc = zeros(1, N_sub, 'single');
            for k = 1:N_sub
                sig_sub = zeros(rcv_num, len);
                Pk_sum = 0;
                for m = 1:rcv_num
                    fft_k = data_fft(m,:) .* sub_masks(k,:);
                    sig_k = ifft(ifftshift(fft_k));
                    sig_sub(m,:) = sig_k;
                    P_k   = mean(abs(sig_k).^2);
                    Pk_sum = Pk_sum + P_k;
                    if P_k > 0
                        sig_k = sig_k / sqrt(P_k);
                    end
                    data_fft_batch(k, m, :) = fftshift(fft(sig_k));
                end
                sub_energy_loc(k) = Pk_sum / rcv_num;
                R_k = (sig_sub * sig_sub');
                cov_mat_real_all(trial, k, :, :) = single(real(R_k));
                cov_mat_imag_all(trial, k, :, :) = single(imag(R_k));
            end

            %% ── 批量DPD ──
            mtr_sub_loc = single(DPD_calculator_gpu_batch(...
                rcvPos, data_fft_batch, init_pos, edge, lamda, fs));

            %% ── 生成实例级标签 ──
            [fc_sorted, sort_idx] = sort(fc_off);
            src_pos = src_pos(sort_idx, :);
            Pt_W    = Pt_W(sort_idx);
            Pt_dBW  = Pt_dBW(sort_idx);
            snr_mat = snr_mat(sort_idx, :);
            fc_off  = fc_sorted;

            band_mask_loc   = zeros(max_src, N_sub, 'single');
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
                    cov = ov_main / B_win;

                    if cov >= thresh
                        band_mask_loc(s, k) = 1;
                    elseif cov > 0 || ov_roll > 0
                        ignore_mask_loc(s, k) = 1;
                    end
                end
            end

            %% ── 直接写入大矩阵 ──
            mtr_sub_all(trial,:,:,:)       = mtr_sub_loc;
            src_count_all(trial)           = int32(n_src);
            band_mask_all(trial,:,:)       = band_mask_loc;
            ignore_mask_all(trial,:,:)     = ignore_mask_loc;
            fc_offset_all(trial, 1:n_src)  = single(fc_off(1:n_src));
            Pt_W_all(trial, 1:n_src)       = single(Pt_W(1:n_src));
            src_pos_all(trial, 1:n_src, :) = single(src_pos(1:n_src, :));
            sub_energy_all(trial,:)        = sub_energy_loc;

            snr_per_src = mean(snr_mat, 2);
            avg_snr_all(trial) = single(min(snr_per_src));

            stat_src_count(n_src + 1) = stat_src_count(n_src + 1) + 1;
            for s = 1:n_src
                stat_band1  = stat_band1  + sum(band_mask_loc(s,:) == 1);
                stat_ignore = stat_ignore + sum(ignore_mask_loc(s,:) == 1);
                stat_band0  = stat_band0  + ...
                    sum(band_mask_loc(s,:) == 0 & ignore_mask_loc(s,:) == 0);
            end
            for s = n_src+1 : max_src
                stat_band0 = stat_band0 + N_sub;
            end

        end  % if n_src == 0

        %% ── 进度 ──
        if mod(trial, 100) == 0
            wait(gpuDevice);
            clearGPU = gpuDevice;
            elapsed = toc(t_start);
            eta     = elapsed / trial * (N_trials - trial);
            fprintf('[%s] Trial %d/%d (%.1f%%) | 耗时 %.0fs | 剩余 %.0fs\n', ...
                set_name, trial, N_trials, 100*trial/N_trials, elapsed, eta);
        end
    end

    %% ── 打印统计 ──
    fprintf('\n===== %s 集生成完毕 =====\n', set_name);
    for c = src_num_range
        fprintf('  %d源: %d 条 (%.1f%%)\n', c, stat_src_count(c+1), ...
                100*stat_src_count(c+1)/N_trials);
    end

    total_labels = stat_band1 + stat_band0 + stat_ignore;
    fprintf('\n子带标签分布 (共 %d 个槽位×子带):\n', total_labels);
    fprintf('  标1: %d (%.1f%%)\n', stat_band1, 100*stat_band1/total_labels);
    fprintf('  标0: %d (%.1f%%)\n', stat_band0, 100*stat_band0/total_labels);
    fprintf('  ignore: %d (%.1f%%)\n', stat_ignore, 100*stat_ignore/total_labels);

    has_src = (src_count_all > 0);
    if any(has_src)
        valid_snr = avg_snr_all(has_src);
        fprintf('SNR范围: [%.1f, %.1f] dB\n', min(valid_snr), max(valid_snr));
    end
    fprintf('耗时 %.1f 秒\n', toc(t_start));

    %% ── 一次性保存为Python可读格式 ──
    save_file = fullfile(runtime.data_dir, sprintf('%s_data.mat', set_name));
    fprintf('正在保存 %s ...\n', save_file);

    save(save_file, ...
        'mtr_sub_all', ...
        'src_count_all', 'band_mask_all', 'ignore_mask_all', ...
        'avg_snr_all', 'sub_energy_all', ...
        'cov_mat_real_all', 'cov_mat_imag_all', ...
        'fc_offset_all', 'Pt_W_all', 'src_pos_all', ...
        'N_sub_val', 'max_src_val', 'num_grid', ...
        'edge_val', 'lamda_val', ...
        'B_win_val', 'B_step_val', 'fs_val', 'symbolRate_val', ...
        'BW_actual_val', 'arfa_val', 'smoke_seed_val', ...
        'sub_f_lo_val', 'sub_f_hi_val', ...
        'thresh_val', 'num_count_classes', ...
        '-v7.3');

    fprintf('已保存至 %s\n', save_file);
    fprintf('  mtr_sub_all:      [%s]\n', num2str(size(mtr_sub_all)));
    fprintf('  src_count_all:    [%s]\n', num2str(size(src_count_all)));
    fprintf('  band_mask_all:    [%s]\n', num2str(size(band_mask_all)));
    fprintf('  ignore_mask_all:  [%s]\n', num2str(size(ignore_mask_all)));

    clear mtr_sub_all src_count_all band_mask_all ignore_mask_all ...
          avg_snr_all sub_energy_all cov_mat_real_all cov_mat_imag_all ...
          fc_offset_all Pt_W_all src_pos_all;
end

fprintf('\n全部完成！总耗时 %.1f 秒\n', toc);


%% ═══════════════════════════════════════════════════════════════
%  频率分配函数
%% ═══════════════════════════════════════════════════════════════
function fc_off = assign_freq_offsets(n_src, fc_min, fc_max, symbolRate)
    fc_off = nan(1, n_src);

    switch n_src
        case 1
            fc_off(1) = fc_min + (fc_max - fc_min) * rand();

        case 2
            r = rand();
            fc_off(1) = fc_min + (fc_max - fc_min) * rand();
            if r < 0.50
                fc_off(2) = fc_off(1);
            elseif r < 0.80
                delta = (0.1 + 0.9*rand()) * symbolRate;
                if rand() < 0.5, delta = -delta; end
                fc_off(2) = clamp_freq(fc_off(1) + delta, fc_min, fc_max);
            else
                fc_off(2) = find_separated_freq(fc_off, 1, fc_min, fc_max, symbolRate);
            end

        case 3
            r = rand();
            if r < 0.35
                fc_base = fc_min + (fc_max - fc_min) * rand();
                fc_off(1) = fc_base; fc_off(2) = fc_base; fc_off(3) = fc_base;
            elseif r < 0.60
                pair = randperm(3, 2); rest = setdiff(1:3, pair);
                fc_base = fc_min + (fc_max - fc_min) * rand();
                fc_off(pair(1)) = fc_base; fc_off(pair(2)) = fc_base;
                delta = (0.1 + 0.9*rand()) * symbolRate;
                if rand() < 0.5, delta = -delta; end
                fc_off(rest) = clamp_freq(fc_base + delta, fc_min, fc_max);
            elseif r < 0.75
                pair = randperm(3, 2); rest = setdiff(1:3, pair);
                fc_base = fc_min + (fc_max - fc_min) * rand();
                fc_off(pair(1)) = fc_base; fc_off(pair(2)) = fc_base;
                fc_off(rest) = find_separated_freq(fc_off, pair, fc_min, fc_max, symbolRate);
            elseif r < 0.90
                pair = randperm(3, 2); rest = setdiff(1:3, pair);
                fc_base = fc_min + (fc_max - fc_min) * rand();
                fc_off(pair(1)) = fc_base;
                delta = (0.1 + 0.9*rand()) * symbolRate;
                if rand() < 0.5, delta = -delta; end
                fc_off(pair(2)) = clamp_freq(fc_base + delta, fc_min, fc_max);
                fc_off(rest) = find_separated_freq(fc_off, pair, fc_min, fc_max, symbolRate);
            else
                fc_off(1) = fc_min + (fc_max - fc_min) * rand();
                fc_off(2) = find_separated_freq(fc_off, 1, fc_min, fc_max, symbolRate);
                fc_off(3) = find_separated_freq(fc_off, [1 2], fc_min, fc_max, symbolRate);
            end
    end
end

function fc_out = find_separated_freq(fc_off, assigned_idx, fc_min, fc_max, symbolRate)
    assigned = fc_off(assigned_idx);
    for iter = 1:50000
        fc_cand = fc_min + (fc_max - fc_min) * rand();
        if all(abs(fc_cand - assigned) >= symbolRate)
            fc_out = fc_cand; return;
        end
    end
    warning('无法找到分离频偏，强制放置');
    fc_out = fc_cand;
end

function f = clamp_freq(f, fc_min, fc_max)
    f = max(fc_min, min(fc_max, f));
end
