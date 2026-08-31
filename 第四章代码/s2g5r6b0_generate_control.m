function report = s2g5r6b0_generate_control(output_dir, n_per_condition, base_seed)
%S2G5R6B0_GENERATE_CONTROL 生成论文第四章4A/4B控制条件的独立诊断IQ。
%
% 该入口只服务S2-G5-R6-B0，不改变main_exp_snr.m/main_exp_dist.m的
% 原始默认行为。每个条件单独保存一个-v7.3 MAT文件，便于流式计算DPD。

if nargin < 1 || isempty(output_dir)
    error('必须显式提供独立输出目录');
end
if nargin < 2 || isempty(n_per_condition)
    n_per_condition = 1000;
end
if nargin < 3 || isempty(base_seed)
    base_seed = 20260830;
end

validateattributes(n_per_condition, {'numeric'}, {'scalar','integer','positive'});
validateattributes(base_seed, {'numeric'}, {'scalar','integer','nonnegative'});
output_dir = char(output_dir);
if exist(output_dir, 'dir')
    entries = dir(output_dir);
    entries = entries(~ismember({entries.name}, {'.','..'}));
    if ~isempty(entries)
        error('拒绝覆盖非空输出目录: %s', output_dir);
    end
else
    mkdir(output_dir);
end

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fileparts(script_dir));
global Txobj Rxobj

cfg.fc = 5800e6;
cfg.arfa = 0.25;
cfg.symbol_rate = 10e6;
cfg.bw_actual = cfg.symbol_rate * (1 + cfg.arfa * 1.2);
cfg.fs = 100e6;
cfg.len = 2^12;
cfg.vc = 299792458;
cfg.max_src = 3;
cfg.noise_dbm = -90;
cfg.noise_w = 10^((cfg.noise_dbm - 30) / 10);
cfg.rcv_radius = 500;
cfg.edge = 2000;
cfg.lamda = 10;
cfg.b_win = 10e6;
cfg.b_step = 5e6;
cfg.n_sub = floor((cfg.fs - cfg.b_win) / cfg.b_step) + 1;
cfg.sub_f_lo = ((0:cfg.n_sub-1) * cfg.b_step - cfg.fs/2);
cfg.sub_f_hi = cfg.sub_f_lo + cfg.b_win;
cfg.hard_threshold = 0.2;
cfg.azimuths = [30, 150, 330];
cfg.min_dist_src2rcv = 200;

angles_rx = (0:3) * 2*pi/4;
Rxobj.Num = 4;
Rxobj.rxId_V = ["001";"002";"003";"004"];
Rxobj.node_pos = [cfg.rcv_radius*cos(angles_rx)', ...
                  cfg.rcv_radius*sin(angles_rx)', zeros(4,1)];
Rxobj.sample_rate = repmat(cfg.fs, 4, 1);
Rxobj.freq_rf = repmat(cfg.fc, 4, 1);
cfg.rcv_pos = Rxobj.node_pos(:,1:2);

Txobj.Num = 1;
Txobj.txId_V = "drone001";
Txobj.freqC_V = cfg.fc;
Txobj.modType_V = "BPSK";
Txobj.multiplexingType_V = "NONE";
Txobj.shapingType_V = "RootRaisedCos";
Txobj.arfa_V = cfg.arfa;
Txobj.symbolRate_V = cfg.symbol_rate;
Txobj.modDepth_V = 1.0;
Txobj.contPhase_V = "cont";
Txobj.nodePos_V = [0,0,0];
Txobj.Gt = 0;
Txobj.antennaType_V = "AntennaX";
Txobj.antennaDeg_V = [0,0];
Txobj.transmitPower_V = 1.0;
Txobj.txTime_V = "2020-06-06 00:00:00:000000";
Txobj.txDuration_V = 0;
Txobj.Bw_V = 0;
cfg.data_len = calc_dataLen(cfg.len, cfg.fs, cfg.symbol_rate);

conditions = struct('experiment', {}, 'n_src', {}, 'parameter_name', {}, ...
                    'parameter_value', {}, 'seed', {}, 'file', {});
snr_values = [-10, -6, 0, 6];
dist_values = [800, 1000];
condition_index = 0;
for n_src = [2, 3]
    for value = snr_values
        condition_index = condition_index + 1;
        conditions(condition_index) = make_condition( ...
            '4A', n_src, 'snr_db', value, base_seed + condition_index, output_dir);
    end
end
for n_src = [2, 3]
    for value = dist_values
        condition_index = condition_index + 1;
        conditions(condition_index) = make_condition( ...
            '4B', n_src, 'distance_m', value, base_seed + condition_index, output_dir);
    end
end

started_all = tic;
entries = repmat(struct('condition_id', '', 'experiment', '', 'n_src', 0, ...
    'parameter_name', '', 'parameter_value', 0, 'seed', 0, ...
    'sample_count', 0, 'path', '', 'size_bytes', 0, 'duration_seconds', 0), ...
    numel(conditions), 1);
for index = 1:numel(conditions)
    cond = conditions(index);
    condition_started = tic;
    fprintf('\n[R6-B0] %s, N=%d, %s=%g, seed=%d, samples=%d\n', ...
        cond.experiment, cond.n_src, cond.parameter_name, ...
        cond.parameter_value, cond.seed, n_per_condition);
    generate_one_condition(cfg, cond, n_per_condition);
    info = dir(cond.file);
    entries(index).condition_id = condition_id(cond);
    entries(index).experiment = cond.experiment;
    entries(index).n_src = cond.n_src;
    entries(index).parameter_name = cond.parameter_name;
    entries(index).parameter_value = cond.parameter_value;
    entries(index).seed = cond.seed;
    entries(index).sample_count = n_per_condition;
    entries(index).path = cond.file;
    entries(index).size_bytes = info.bytes;
    entries(index).duration_seconds = toc(condition_started);
end

report = struct();
report.status = 'PASS';
report.gate = 'S2-G5-R6-B0';
report.scope = 'paper_condition_control_diagnostic_not_frozen_test';
report.base_seed = base_seed;
report.samples_per_condition = n_per_condition;
report.condition_count = numel(conditions);
report.total_samples = n_per_condition * numel(conditions);
report.conditions = entries;
report.fixed_parameters = struct( ...
    'fc_hz', cfg.fc, 'fs_hz', cfg.fs, 'observation_length', cfg.len, ...
    'symbol_rate_hz', cfg.symbol_rate, 'bw_actual_hz', cfg.bw_actual, ...
    'rolloff', cfg.arfa, 'receiver_radius_m', cfg.rcv_radius, ...
    'receiver_count', 4, 'fine_grid_step_m', cfg.lamda, ...
    'azimuths_deg', cfg.azimuths, 'cochannel_offsets_hz', [0,0,0], ...
    'hard19_threshold', cfg.hard_threshold);
report.duration_seconds = toc(started_all);
report.test_executed = false;
report.training_executed = false;

report_path = fullfile(output_dir, 'matlab_generation_report.json');
fid = fopen(report_path, 'w', 'n', 'UTF-8');
if fid < 0
    error('无法创建报告: %s', report_path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(report, PrettyPrint=true));
clear cleanup;
fprintf('\n[R6-B0] 完成: %d条件, %d样本, %.1fs\n', ...
    report.condition_count, report.total_samples, report.duration_seconds);
end


function cond = make_condition(experiment, n_src, parameter_name, parameter_value, seed, output_dir)
cond.experiment = experiment;
cond.n_src = n_src;
cond.parameter_name = parameter_name;
cond.parameter_value = parameter_value;
cond.seed = seed;
if strcmp(parameter_name, 'snr_db')
    if parameter_value < 0
        value_text = sprintf('m%02d', abs(parameter_value));
    else
        value_text = sprintf('p%02d', parameter_value);
    end
    filename = sprintf('%s_K%d_snr_%s.mat', experiment, n_src, value_text);
else
    filename = sprintf('%s_K%d_dist_%04d.mat', experiment, n_src, parameter_value);
end
cond.file = fullfile(output_dir, filename);
end


function value = condition_id(cond)
if strcmp(cond.parameter_name, 'snr_db')
    value = sprintf('%s_K%d_SNR_%+d', cond.experiment, cond.n_src, cond.parameter_value);
else
    value = sprintf('%s_K%d_DIST_%d', cond.experiment, cond.n_src, cond.parameter_value);
end
end


function generate_one_condition(cfg, cond, n_samples)
global Txobj
if exist(cond.file, 'file')
    error('拒绝覆盖已有MAT文件: %s', cond.file);
end
rng(cond.seed, 'twister');
n_src = cond.n_src;
rcv_num = size(cfg.rcv_pos, 1);
fc_off = zeros(1, n_src);
P_noise_inband = cfg.noise_w * (cfg.bw_actual / cfg.fs);

sig_rcv_real_all = zeros(n_samples, rcv_num, cfg.len, 'single');
sig_rcv_imag_all = zeros(n_samples, rcv_num, cfg.len, 'single');
src_count_all = repmat(int32(n_src), n_samples, 1);
fc_offset_all = zeros(n_samples, cfg.max_src, 'single');
Pt_W_all = zeros(n_samples, cfg.max_src, 'single');
src_pos_all = zeros(n_samples, cfg.max_src, 2, 'single');
symbolRate_all = zeros(n_samples, cfg.max_src, 'single');
BW_actual_all = zeros(n_samples, cfg.max_src, 'single');
avg_snr_all = zeros(n_samples, 1, 'single');
snr_param_all = nan(n_samples, 1, 'single');
dist_param_all = nan(n_samples, 1, 'single');
rotation_rad_all = zeros(n_samples, 1, 'single');
sample_id_all = int32((0:n_samples-1)');

started = tic;
for trial = 1:n_samples
    if strcmp(cond.experiment, '4A')
        distance_m = 800;
        target_snr_dB = cond.parameter_value;
        rotation_rad = 0;
    else
        distance_m = cond.parameter_value;
        target_snr_dB = 0;
        while true
            rotation_rad = 2*pi*rand();
            candidate = source_positions(cfg.azimuths, n_src, distance_m, rotation_rad);
            if nearest_receiver_distance(candidate, cfg.rcv_pos) >= cfg.min_dist_src2rcv
                break;
            end
        end
    end
    src_pos = source_positions(cfg.azimuths, n_src, distance_m, rotation_rad);

    PL_dB_mat = zeros(n_src, rcv_num);
    for s = 1:n_src
        for m = 1:rcv_num
            PL_dB_mat(s,m) = PL_free(cfg.fc, ...
                norm(src_pos(s,:) - cfg.rcv_pos(m,:)), 0, 0);
        end
    end
    PL_avg_dB = mean(PL_dB_mat, 2);
    Pt_W = zeros(1, n_src);
    target_snr_lin = 10^(target_snr_dB/10);
    for s = 1:n_src
        Pt_W(s) = target_snr_lin * P_noise_inband * 10^(PL_avg_dB(s)/10);
    end

    sig_pool = zeros(n_src, cfg.len);
    for s = 1:n_src
        Txobj.symbolRate_V = cfg.symbol_rate;
        [~, tmp] = Gen_basesig(cfg.data_len, cfg.fs, Txobj.txId_V);
        tmp = tmp(1:cfg.len);
        sig_pool(s,:) = tmp / sqrt(mean(abs(tmp).^2));
    end

    sig_rcv_accum = zeros(rcv_num, cfg.len);
    snr_actual = zeros(n_src, rcv_num);
    t_vec = (0:cfg.len-1) / cfg.fs;
    for s = 1:n_src
        baseSig = sig_pool(s,:) .* exp(1j*2*pi*fc_off(s)*t_vec);
        baseSig = baseSig / sqrt(mean(abs(baseSig).^2));
        for m = 1:rcv_num
            tau_m = norm(src_pos(s,:) - cfg.rcv_pos(m,:)) / cfg.vc;
            sig_del = apply_delay_fd(baseSig, tau_m, cfg.fs);
            Pr_W = Pt_W(s) * 10^(-PL_dB_mat(s,m)/10);
            sig_rcv_accum(m,:) = sig_rcv_accum(m,:) + sqrt(Pr_W)*sig_del;
            snr_actual(s,m) = 10*log10(Pr_W/P_noise_inband);
        end
    end

    sig_rcv = zeros(rcv_num, cfg.len);
    for m = 1:rcv_num
        noise = sqrt(cfg.noise_w/2) * ...
            (randn(1,cfg.len) + 1j*randn(1,cfg.len));
        sig_rcv(m,:) = sig_rcv_accum(m,:) + noise;
    end

    sig_rcv_real_all(trial,:,:) = single(real(sig_rcv));
    sig_rcv_imag_all(trial,:,:) = single(imag(sig_rcv));
    fc_offset_all(trial,1:n_src) = single(fc_off);
    Pt_W_all(trial,1:n_src) = single(Pt_W);
    src_pos_all(trial,1:n_src,:) = single(src_pos);
    symbolRate_all(trial,1:n_src) = single(cfg.symbol_rate);
    BW_actual_all(trial,1:n_src) = single(cfg.bw_actual);
    avg_snr_all(trial) = single(mean(mean(snr_actual,2)));
    snr_param_all(trial) = single(target_snr_dB);
    dist_param_all(trial) = single(distance_m);
    rotation_rad_all(trial) = single(rotation_rad);

    if mod(trial, 100) == 0 || trial == n_samples
        elapsed = toc(started);
        fprintf('  %s %d/%d (%.1f%%), elapsed %.1fs, ETA %.1fs\n', ...
            condition_id(cond), trial, n_samples, 100*trial/n_samples, ...
            elapsed, elapsed/trial*(n_samples-trial));
    end
end

rcv_pos_val = single(cfg.rcv_pos);
BW_actual_val = single(cfg.bw_actual);
N_power_W_val = single(cfg.noise_w);
N_power_dBm_val = single(cfg.noise_dbm);
exp_n_src_val = int32(n_src);
exp_n_per_condition_val = int32(n_samples);
exp_name_val = cond.experiment;
condition_id_val = condition_id(cond);
parameter_name_val = cond.parameter_name;
parameter_value_val = single(cond.parameter_value);
random_seed_val = int32(cond.seed);
N_sub_val = int32(cfg.n_sub);
max_src_val = int32(cfg.max_src);
edge_val = single(cfg.edge);
lamda_val = single(cfg.lamda);
B_win_val = single(cfg.b_win);
B_step_val = single(cfg.b_step);
fs_val = single(cfg.fs);
symbolRate_val = single(cfg.symbol_rate);
sub_f_lo_val = single(cfg.sub_f_lo);
sub_f_hi_val = single(cfg.sub_f_hi);
thresh_val = single(cfg.hard_threshold);
num_count_classes = int32(cfg.max_src+1);
exp_azimuths_val = single(cfg.azimuths(1:n_src));

save(cond.file, ...
    'src_count_all', 'avg_snr_all', 'fc_offset_all', 'Pt_W_all', ...
    'src_pos_all', 'symbolRate_all', 'BW_actual_all', ...
    'sig_rcv_real_all', 'sig_rcv_imag_all', ...
    'snr_param_all', 'dist_param_all', 'rotation_rad_all', 'sample_id_all', ...
    'rcv_pos_val', 'BW_actual_val', 'N_power_W_val', 'N_power_dBm_val', ...
    'exp_n_src_val', 'exp_n_per_condition_val', 'exp_name_val', ...
    'condition_id_val', 'parameter_name_val', 'parameter_value_val', ...
    'random_seed_val', 'exp_azimuths_val', ...
    'N_sub_val', 'max_src_val', 'edge_val', 'lamda_val', ...
    'B_win_val', 'B_step_val', 'fs_val', 'symbolRate_val', ...
    'sub_f_lo_val', 'sub_f_hi_val', 'thresh_val', 'num_count_classes', ...
    '-v7.3');
end


function positions = source_positions(azimuths, n_src, distance_m, rotation_rad)
positions = zeros(n_src, 2);
for s = 1:n_src
    angle = deg2rad(azimuths(s)) + rotation_rad;
    positions(s,:) = [distance_m*cos(angle), distance_m*sin(angle)];
end
end


function result = nearest_receiver_distance(src_pos, rcv_pos)
result = inf;
for s = 1:size(src_pos,1)
    for m = 1:size(rcv_pos,1)
        result = min(result, norm(src_pos(s,:) - rcv_pos(m,:)));
    end
end
end
