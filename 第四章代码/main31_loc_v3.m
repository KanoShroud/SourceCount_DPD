%% main31_loc_v3.m  多源信源检测 + 保存IQ信号 (第四章定位训练数据 v3)
%
% 基于 main31_loc_v2.m，改动：
%   1. 每源独立 symbolRate (均匀 [500kHz, 15.4MHz])
%   2. 每源独立频偏 (保证所有源频率重叠)
%   3. SIR 扩展: 源间最大功率比 10dB
%   4. SNR 按 BW_union 定义: SNR_s = Pr_s / (N0 × BW_union)
%   5. 新增 min_dist_src2rcv = 150m
%   6. 保存 symbolRate_all, BW_actual_all
%
% SNR定义：
%   SNR_s = Pr_s_avg / (N_power_W × BW_union / fs)
%   BW_union = 所有源频率范围的并集宽度
%
clc; clear; close all;
tic
global Txobj Rxobj

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fileparts(script_dir));
runtime = gate0_runtime('chapter4', mfilename);

if runtime.is_smoke
    fprintf('[Gate2] smoke 模式跳过主循环未使用的 MATLAB 并行池。\n');
else
    if isempty(gcp('nocreate'))
        parpool('local');
    end
end

%% ═══════════════════════════════════════
%  参数配置区
%% ═══════════════════════════════════════
set_list = {'train', 'val', 'test'};
if runtime.is_smoke
    trials_list = [4, 2, 2];
    random_seed_val = int32(20260821);

    trials_env = strtrim(getenv('SOURCECOUNT_CH4_TRIALS_LIST'));
    if ~isempty(trials_env)
        if isempty(regexp(trials_env, '^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$', 'once'))
            error(['SOURCECOUNT_CH4_TRIALS_LIST 必须是 3 个逗号分隔的正整数，' ...
                '例如 1024,256,256。当前值: %s'], trials_env);
        end
        parsed_trials = sscanf(trials_env, '%f,%f,%f');
        trials_cap = [2048, 512, 512];
        if numel(parsed_trials) ~= 3 || any(~isfinite(parsed_trials)) || ...
                any(parsed_trials <= 0) || any(parsed_trials ~= floor(parsed_trials))
            error(['SOURCECOUNT_CH4_TRIALS_LIST 必须是 3 个逗号分隔的正整数，' ...
                '例如 1024,256,256。当前值: %s'], trials_env);
        end
        trials_list = double(parsed_trials(:)).';
        if any(trials_list > trials_cap)
            error(['smoke trials 超出 Gate 3 硬上限 [%d,%d,%d]，当前为 ' ...
                '[%d,%d,%d]。'], trials_cap, trials_list);
        end
        fprintf('[Gate3] smoke trials 覆盖为 [%d,%d,%d]。\n', trials_list);
    end

    seed_env = strtrim(getenv('SOURCECOUNT_CH4_RANDOM_SEED'));
    if ~isempty(seed_env)
        if isempty(regexp(seed_env, '^\d+$', 'once'))
            error(['SOURCECOUNT_CH4_RANDOM_SEED 必须是 [0,%d] 内的整数，' ...
                '当前值: %s'], intmax('int32'), seed_env);
        end
        parsed_seed = str2double(seed_env);
        if ~isfinite(parsed_seed) || parsed_seed < 0 || ...
                parsed_seed ~= floor(parsed_seed) || parsed_seed > double(intmax('int32'))
            error(['SOURCECOUNT_CH4_RANDOM_SEED 必须是 [0,%d] 内的整数，' ...
                '当前值: %s'], intmax('int32'), seed_env);
        end
        random_seed_val = int32(parsed_seed);
        fprintf('[Gate3] smoke 随机种子覆盖为 %d。\n', random_seed_val);
    end
    rng(double(random_seed_val), 'twister');
else
    trials_list = [40000, 5000, 5000];
    random_seed_val = int32(-1);
end
runtime_mode_val = runtime.mode;
trials_list_val = int32(trials_list);

fc          = 5800e6;
arfa_V      = 0.25;
fs          = 100e6;

%% ── [v3 改动] 带宽随机化 ──
symbolRate_min = 2e6;     % 最小符号率 500kHz
symbolRate_max = 20e6;    % 最大符号率 15.4MHz → BW_max ≈ 20MHz

B_win       = 10e6;
B_step      = 5e6;
N_sub       = floor((fs - B_win) / B_step) + 1;   % = 19

len         = 2^12;
f_axis      = (-len/2 : len/2-1) * (fs/len);

%% ── 信源场景配置 ──
max_src         = 3;
src_num_range   = [0, 1, 2, 3];
src_num_weights = [0.00, 0.00, 0.50, 0.50];

%% ── [v3 改动] 功率配置 ──
snr_range_dB     = [-10, 10];    % 各源 SNR 范围 (dB), 相对 N0*BW_union
max_power_ratio_dB = 3;         % 源间最大功率比 (dB), 即 |SIR| ≤ 10dB

N_power_dBm = -90;
N_power_dBW = N_power_dBm - 30;
N_power_W   = 10^(N_power_dBW / 10);

%% ── 空间配置 ──
dist_range         = [100, 1000];   % 下限100m避免小半径无法满足源间距
dist_jitter_ratio  = 0.1;
min_dist_src2src   = 150;
min_dist_src2rcv   = 150;          % [v3 新增] 源到最近接收站最小距离

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
Txobj.symbolRate_V       = 10e6;      % 会被逐源覆盖
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
sub_f_lo  = zeros(1, N_sub);
sub_f_hi  = zeros(1, N_sub);
for k = 1:N_sub
    sub_f_lo(k) = (k-1)*B_step - fs/2;
    sub_f_hi(k) = (k-1)*B_step - fs/2 + B_win;
end

fprintf('═══════════════════════════════════════\n');
fprintf('  main31_loc_v3  (独立带宽+频偏+SIR扩展)\n');
fprintf('═══════════════════════════════════════\n');
fprintf('采样率: %.0fMHz  子带: N_sub=%d\n', fs/1e6, N_sub);
fprintf('symbolRate: [%.0fkHz, %.1fMHz] 均匀\n', symbolRate_min/1e3, symbolRate_max/1e6);
fprintf('BW_actual:  [%.0fkHz, %.1fMHz]\n', ...
    symbolRate_min*(1+arfa_V*1.2)/1e3, symbolRate_max*(1+arfa_V*1.2)/1e6);
fprintf('接收站: %d个, R=%.0fm\n', rcv_num, R_rcv);
fprintf('SNR范围: [%+d, %+d] dB (相对 N0×BW_union)\n', snr_range_dB(1), snr_range_dB(2));
fprintf('最大功率比: %d dB (|SIR| ≤ %ddB)\n', max_power_ratio_dB, max_power_ratio_dB);
fprintf('源间距: ≥%dm  源站距: ≥%dm\n', min_dist_src2src, min_dist_src2rcv);
fprintf('噪底: %.0f dBm (固定)\n', N_power_dBm);
fprintf('max_src=%d  频偏随机(保证重叠)\n\n', max_src);

%% ═══════════════════════════════════════
%  保存配置参数
%% ═══════════════════════════════════════
N_sub_val         = int32(N_sub);
max_src_val       = int32(max_src);
edge_val          = single(edge);
lamda_val         = single(lamda);
B_win_val         = single(B_win);
B_step_val        = single(B_step);
fs_val            = single(fs);
sub_f_lo_val      = single(sub_f_lo);
sub_f_hi_val      = single(sub_f_hi);
thresh_val        = single(thresh);
num_count_classes = int32(max_src + 1);

%% ═══════════════════════════════════════
%  三轮循环
%% ═══════════════════════════════════════
for si = 1:length(set_list)

    set_name = set_list{si};
    N_trials = trials_list(si);

    fprintf('\n========== 开始生成 %s 集 (%d 条) ==========\n', set_name, N_trials);
    if N_trials == 0, continue; end

    %% ── 信源数量分配 ──
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

    %% ── 预分配 ──
    src_count_all    = zeros(N_trials, 1, 'int32');
    band_mask_all    = zeros(N_trials, max_src, N_sub, 'single');
    ignore_mask_all  = zeros(N_trials, max_src, N_sub, 'single');
    avg_snr_all      = zeros(N_trials, 1, 'single');
    fc_offset_all    = zeros(N_trials, max_src, 'single');
    Pt_W_all         = zeros(N_trials, max_src, 'single');
    src_pos_all      = zeros(N_trials, max_src, 2, 'single');
    sig_rcv_real_all = zeros(N_trials, rcv_num, len, 'single');
    sig_rcv_imag_all = zeros(N_trials, rcv_num, len, 'single');
    symbolRate_all   = zeros(N_trials, max_src, 'single');    % [v3 新增]
    BW_actual_all    = zeros(N_trials, max_src, 'single');    % [v3 新增]

    %% ── 统计 ──
    stat_src_count = zeros(1, max_src + 1);
    stat_band1  = 0;
    stat_band0  = 0;
    stat_ignore = 0;
    stat_bw_union = [];

    t_start = tic;

    for trial = 1:N_trials

        n_src = src_num_seq(trial);

        if n_src == 0
            %% ════════════════════════════════
            %  0源：纯噪声
            %% ════════════════════════════════
            sig_rcv = zeros(rcv_num, len);
            for m = 1:rcv_num
                noise = sqrt(N_power_W/2) * (randn(1,len) + 1j*randn(1,len));
                sig_rcv(m,:) = noise;
            end

            sig_rcv_real_all(trial,:,:) = single(real(sig_rcv));
            sig_rcv_imag_all(trial,:,:) = single(imag(sig_rcv));
            src_count_all(trial)        = int32(0);
            avg_snr_all(trial)          = single(-999);
            stat_src_count(1) = stat_src_count(1) + 1;
            stat_band0 = stat_band0 + max_src * N_sub;

        else
            %% ════════════════════════════════
            %  有源
            %% ════════════════════════════════

            %% ── [v3] 每源独立 symbolRate（量化到50kHz步长，确保resample整数比）──
            symbolRate_s = zeros(1, n_src);
            BW_actual_s  = zeros(1, n_src);
            for s = 1:n_src
                sr_raw = symbolRate_min + (symbolRate_max - symbolRate_min) * rand();
                symbolRate_s(s) = round(sr_raw / 50e3) * 50e3;  % 量化到50kHz
                symbolRate_s(s) = max(symbolRate_s(s), symbolRate_min);
                BW_actual_s(s)  = symbolRate_s(s) * (1 + arfa_V * 1.2);
            end

            %% ── [v3] 每源独立频偏（保证所有源重叠）──
            fc_off = zeros(1, n_src);

            % 源1: 随机频偏
            fc_off_min_1 = -(fs/2 - BW_actual_s(1)/2);
            fc_off_max_1 =  (fs/2 - BW_actual_s(1)/2);
            fc_off(1) = fc_off_min_1 + (fc_off_max_1 - fc_off_min_1) * rand();

            % 源2+: 偏移但保证与组重叠
            for s = 2:n_src
                % 当前组的频率范围
                group_lo = min(fc_off(1:s-1) - BW_actual_s(1:s-1)/2);
                group_hi = max(fc_off(1:s-1) + BW_actual_s(1:s-1)/2);

                % 新源必须与组重叠:
                %   fc_off(s) - BW_s/2 < group_hi  且
                %   fc_off(s) + BW_s/2 > group_lo
                % → group_lo - BW_s/2 < fc_off(s) < group_hi + BW_s/2
                lo_bound = group_lo - BW_actual_s(s)/2 + 1e3;   % +1kHz 保证严格重叠
                hi_bound = group_hi + BW_actual_s(s)/2 - 1e3;

                % 钳位到合法频率范围
                fc_off_min_s = -(fs/2 - BW_actual_s(s)/2);
                fc_off_max_s =  (fs/2 - BW_actual_s(s)/2);
                lo_bound = max(lo_bound, fc_off_min_s);
                hi_bound = min(hi_bound, fc_off_max_s);

                if lo_bound >= hi_bound
                    fc_off(s) = fc_off(1);   % fallback: 和源1同频
                else
                    fc_off(s) = lo_bound + (hi_bound - lo_bound) * rand();
                end
            end

            %% ── [v3] 计算 BW_union ──
            freq_lo_all = fc_off(1:n_src) - BW_actual_s(1:n_src)/2;
            freq_hi_all = fc_off(1:n_src) + BW_actual_s(1:n_src)/2;
            BW_union = max(freq_hi_all) - min(freq_lo_all);
            N_inband = N_power_W * (BW_union / fs);
            stat_bw_union = [stat_bw_union, BW_union];

            %% ── [v3] 生成基带信号（每源独立带宽）──
            sig_pool = zeros(n_src, len);
            for s = 1:n_src
                Txobj.symbolRate_V = symbolRate_s(s);
                [~] = evalc('dataLen_s = calc_dataLen(len, fs, symbolRate_s(s))');
                [~, tmp] = Gen_basesig(dataLen_s, fs, Txobj.txId_V);
                tmp = tmp(1:len);
                sig_pool(s, :) = tmp / sqrt(mean(abs(tmp).^2));
            end

            %% ── 随机信源位置 ──
            % 最小半径: 确保圆上能放 n_src 个间距 ≥ min_dist 的点
            if n_src >= 2
                r_min_geom = min_dist_src2src / (2 * sin(pi / n_src)) + 10;
            else
                r_min_geom = 0;
            end
            dist_min_base = max(dist_range(1), r_min_geom);
            dist_base = dist_min_base + (dist_range(2) - dist_min_base) * rand();
            dist_lo = max(dist_range(1), dist_base * (1 - dist_jitter_ratio));
            dist_hi = min(dist_range(2), dist_base * (1 + dist_jitter_ratio));
            src_pos = gen_multi_source_pos_v2(...
                rcvPos, n_src, [dist_lo, dist_hi], min_dist_src2src);

            %% ── [v3] 检查源到接收站最小距离 ──
            max_retries = 50;
            for retry = 1:max_retries
                too_close = false;
                for s = 1:n_src
                    for m = 1:rcv_num
                        if norm(src_pos(s,:) - rcvPos(m,:)) < min_dist_src2rcv
                            too_close = true;
                            break;
                        end
                    end
                    if too_close, break; end
                end
                if ~too_close, break; end
                % 重新生成
                src_pos = gen_multi_source_pos_v2(...
                    rcvPos, n_src, [dist_lo, dist_hi], min_dist_src2src);
            end

            %% ── 路径损耗 ──
            PL_dB_mat = zeros(n_src, rcv_num);
            avg_gain  = zeros(1, n_src);
            for s = 1:n_src
                for m = 1:rcv_num
                    PL_dB_mat(s,m) = PL_free(fc, ...
                        norm(src_pos(s,:) - rcvPos(m,:)), 0, 0);
                end
                avg_gain(s) = mean(10.^(-PL_dB_mat(s,:)/10));
            end

            %% ── [v3] 按 BW_union SNR 反推发射功率 ──
            % 源1 = 最弱源, 源2+ 比源1强 0~max_power_ratio dB
            weak_snr_dB = snr_range_dB(1) + diff(snr_range_dB) * rand();
            snr_each_dB = zeros(1, n_src);
            snr_each_dB(1) = weak_snr_dB;
            for s = 2:n_src
                snr_each_dB(s) = weak_snr_dB + max_power_ratio_dB * rand();
            end
            snr_each_dB = min(snr_each_dB, snr_range_dB(2));

            Pt_W = zeros(1, n_src);
            for s = 1:n_src
                Pt_W(s) = 10^(snr_each_dB(s)/10) * N_inband / avg_gain(s);
            end

            %% ── 构造接收信号 ──
            sig_rcv_accum = zeros(rcv_num, len);
            snr_mat       = zeros(n_src, rcv_num);
            t_vec         = (0:len-1) / fs;

            for s = 1:n_src
                baseSig = sig_pool(s, :);
                baseSig = baseSig .* exp(1j * 2*pi * fc_off(s) * t_vec);
                baseSig = baseSig / sqrt(mean(abs(baseSig).^2));

                for m = 1:rcv_num
                    tau_m   = norm(src_pos(s,:) - rcvPos(m,:)) / vc;
                    sig_del = apply_delay_fd(baseSig, tau_m, fs);
                    Pr_W    = Pt_W(s) * 10^(-PL_dB_mat(s,m)/10);
                    sig_rcv_accum(m,:) = sig_rcv_accum(m,:) + sqrt(Pr_W) * sig_del;
                    snr_mat(s,m) = 10*log10(Pr_W / N_inband);
                end
            end

            %% ── 加噪声 ──
            sig_rcv = zeros(rcv_num, len);
            for m = 1:rcv_num
                noise = sqrt(N_power_W/2) * (randn(1,len) + 1j*randn(1,len));
                sig_rcv(m,:) = sig_rcv_accum(m,:) + noise;
            end

            %% ── 保存 IQ ──
            sig_rcv_real_all(trial,:,:) = single(real(sig_rcv));
            sig_rcv_imag_all(trial,:,:) = single(imag(sig_rcv));

            %% ── [v3] 生成子带标签（逐源带宽）──
            [fc_sorted, sort_idx] = sort(fc_off);
            src_pos     = src_pos(sort_idx, :);
            Pt_W        = Pt_W(sort_idx);
            snr_mat     = snr_mat(sort_idx, :);
            snr_each_dB = snr_each_dB(sort_idx);
            fc_off      = fc_sorted;
            symbolRate_s = symbolRate_s(sort_idx);
            BW_actual_s  = BW_actual_s(sort_idx);

            band_mask_loc   = zeros(max_src, N_sub, 'single');
            ignore_mask_loc = zeros(max_src, N_sub, 'single');

            for s = 1:n_src
                % [v3] 使用逐源带宽
                mainlobe_lo = fc_off(s) - symbolRate_s(s)/2;
                mainlobe_hi = fc_off(s) + symbolRate_s(s)/2;
                rolloff_lo  = fc_off(s) - BW_actual_s(s)/2;
                rolloff_hi  = fc_off(s) + BW_actual_s(s)/2;

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

            %% ── 写入大矩阵 ──
            src_count_all(trial)           = int32(n_src);
            band_mask_all(trial,:,:)       = band_mask_loc;
            ignore_mask_all(trial,:,:)     = ignore_mask_loc;
            fc_offset_all(trial, 1:n_src)  = single(fc_off(1:n_src));
            Pt_W_all(trial, 1:n_src)       = single(Pt_W(1:n_src));
            src_pos_all(trial, 1:n_src, :) = single(src_pos(1:n_src, :));
            symbolRate_all(trial, 1:n_src) = single(symbolRate_s(1:n_src));   % [v3]
            BW_actual_all(trial, 1:n_src)  = single(BW_actual_s(1:n_src));    % [v3]

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
        if mod(trial, 2000) == 0
            elapsed = toc(t_start);
            eta     = elapsed / trial * (N_trials - trial);
            if n_src > 0
                bw_str = '';
                for s = 1:n_src
                    if s > 1, bw_str = [bw_str '+']; end
                    if symbolRate_s(s) >= 1e6
                        bw_str = [bw_str sprintf('%.1fM', symbolRate_s(s)/1e6)];
                    else
                        bw_str = [bw_str sprintf('%.0fk', symbolRate_s(s)/1e3)];
                    end
                end
                power_ratio = max(snr_each_dB) - min(snr_each_dB);
                fprintf('[%s] %d/%d (%.1f%%) | %.0fs | 剩余%.0fs | BW=%s SNR=[%s] PR=%.1fdB\n', ...
                    set_name, trial, N_trials, 100*trial/N_trials, elapsed, eta, ...
                    bw_str, num2str(snr_each_dB, '%.1f '), power_ratio);
            else
                fprintf('[%s] %d/%d (%.1f%%) | %.0fs | 剩余%.0fs | 0源\n', ...
                    set_name, trial, N_trials, 100*trial/N_trials, elapsed, eta);
            end
        end
    end

    %% ── 统计 ──
    fprintf('\n===== %s 集生成完毕 =====\n', set_name);
    for c = src_num_range
        fprintf('  %d源: %d 条 (%.1f%%)\n', c, stat_src_count(c+1), ...
                100*stat_src_count(c+1)/N_trials);
    end

    total_labels = stat_band1 + stat_band0 + stat_ignore;
    fprintf('\n子带标签分布 (共 %d 个):\n', total_labels);
    fprintf('  标1: %d (%.1f%%)\n', stat_band1, 100*stat_band1/total_labels);
    fprintf('  标0: %d (%.1f%%)\n', stat_band0, 100*stat_band0/total_labels);
    fprintf('  ignore: %d (%.1f%%)\n', stat_ignore, 100*stat_ignore/total_labels);

    has_src = (src_count_all > 0);
    if any(has_src)
        valid_snr = avg_snr_all(has_src);
        fprintf('SNR范围 (最弱源): [%.1f, %.1f] dB\n', min(valid_snr), max(valid_snr));
    end
    if ~isempty(stat_bw_union)
        fprintf('BW_union: [%.1fkHz, %.1fMHz], mean=%.1fMHz\n', ...
            min(stat_bw_union)/1e3, max(stat_bw_union)/1e6, mean(stat_bw_union)/1e6);
    end

    iq_size_GB = N_trials * rcv_num * len * 4 * 2 / 1e9;
    fprintf('IQ数据量: %.2f GB\n', iq_size_GB);
    fprintf('耗时 %.1f 秒\n', toc(t_start));

    %% ── 保存 ──
    save_file = fullfile(runtime.data_dir, sprintf('%s_data.mat', set_name));
    fprintf('正在保存 %s ...\n', save_file);

    save(save_file, ...
        'src_count_all', 'band_mask_all', 'ignore_mask_all', ...
        'avg_snr_all', ...
        'fc_offset_all', 'Pt_W_all', 'src_pos_all', ...
        'sig_rcv_real_all', 'sig_rcv_imag_all', ...
        'symbolRate_all', 'BW_actual_all', ...
        'N_sub_val', 'max_src_val', ...
        'edge_val', 'lamda_val', ...
        'B_win_val', 'B_step_val', 'fs_val', ...
        'sub_f_lo_val', 'sub_f_hi_val', ...
        'thresh_val', 'num_count_classes', ...
        'runtime_mode_val', 'random_seed_val', 'trials_list_val', ...
        '-v7.3');

    fprintf('已保存至 %s\n', save_file);
    fprintf('  sig_rcv_real_all: [%s]\n', num2str(size(sig_rcv_real_all)));
    fprintf('  symbolRate_all:   [%s]\n', num2str(size(symbolRate_all)));
    fprintf('  BW_actual_all:    [%s]\n', num2str(size(BW_actual_all)));

    clear src_count_all band_mask_all ignore_mask_all ...
          avg_snr_all fc_offset_all Pt_W_all src_pos_all ...
          sig_rcv_real_all sig_rcv_imag_all ...
          symbolRate_all BW_actual_all;
    stat_bw_union = [];
end

fprintf('\n全部完成！总耗时 %.1f 秒\n', toc);
