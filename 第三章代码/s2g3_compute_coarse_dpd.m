function s2g3_compute_coarse_dpd(input_mat, output_mat, max_samples)
%S2G3_COMPUTE_COARSE_DPD 从第四章共享IQ计算第三章19子带粗DPD。
%
% 本函数是S2-G3受控入口，不修改main30.m或原数据生成流程。输入为
% main31_loc_v3.m生成的v7.3 MAT文件，输出shape和字段满足第三章
% SourceDetectionDataset的读取约定。

if nargin < 3 || isempty(max_samples)
    max_samples = inf;
end

input_mat = char(string(input_mat));
output_mat = char(string(output_mat));
if ~isfile(input_mat)
    error('S2G3:MissingInput', '输入MAT不存在: %s', input_mat);
end
if isfile(output_mat)
    error('S2G3:RefuseOverwrite', '拒绝覆盖已有输出: %s', output_mat);
end
if ~(isscalar(max_samples) && isfinite(max_samples) && max_samples >= 1)
    if ~isinf(max_samples)
        error('S2G3:InvalidMaxSamples', 'max_samples必须为正整数或Inf。');
    end
end

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir, fileparts(script_dir));

required = { ...
    'sig_rcv_real_all', 'sig_rcv_imag_all', 'src_count_all', ...
    'band_mask_all', 'ignore_mask_all', 'avg_snr_all', ...
    'fc_offset_all', 'src_pos_all', 'symbolRate_all', 'BW_actual_all', ...
    'N_sub_val', 'max_src_val', 'B_win_val', 'B_step_val', 'fs_val', ...
    'sub_f_lo_val', 'sub_f_hi_val', 'thresh_val', 'num_count_classes'};
data = load(input_mat, required{:});
for idx = 1:numel(required)
    if ~isfield(data, required{idx})
        error('S2G3:MissingField', '输入MAT缺少字段: %s', required{idx});
    end
end

sig_size = size(data.sig_rcv_real_all);
if ~isequal(sig_size, size(data.sig_rcv_imag_all)) || numel(sig_size) ~= 3
    error('S2G3:IQShape', 'IQ实部/虚部shape不一致或不是三维。');
end
N_total = sig_size(1);
rcv_num = sig_size(2);
len = sig_size(3);
N = min(N_total, floor(max_samples));
N_sub = double(data.N_sub_val);
max_src = double(data.max_src_val);
fs = double(data.fs_val);
if N_sub ~= 19 || rcv_num ~= 4 || len ~= 4096 || max_src ~= 3 || fs ~= 100e6
    error('S2G3:ContractMismatch', ...
        '输入契约不符: N_sub=%d rcv=%d len=%d max_src=%d fs=%.0f。', ...
        N_sub, rcv_num, len, max_src, fs);
end
if size(data.src_count_all, 1) < N
    error('S2G3:SampleCount', '标签样本数少于IQ样本数。');
end

sub_f_lo = double(reshape(data.sub_f_lo_val, 1, []));
sub_f_hi = double(reshape(data.sub_f_hi_val, 1, []));
if numel(sub_f_lo) ~= N_sub || numel(sub_f_hi) ~= N_sub
    error('S2G3:SubbandShape', '子带边界数量不等于N_sub。');
end
f_axis = (-len/2 : len/2-1) * (fs/len);
sub_masks = false(N_sub, len);
for k = 1:N_sub
    sub_masks(k, :) = (f_axis >= sub_f_lo(k)) & (f_axis < sub_f_hi(k));
end

R_rcv = 500;
angles_rx = (0:rcv_num-1) * 2*pi/rcv_num;
rcvPos = [R_rcv*cos(angles_rx)', R_rcv*sin(angles_rx)'];
init_pos = [0, 0];
edge = 2000;
lamda = 50;
num_grid = round(2 * edge / lamda) + 1;

gpu_info = gpuDevice(1); %#ok<NASGU>
mtr_sub_all = zeros(N, N_sub, num_grid, num_grid, 'single');
sample_idx_all = int32((0:N-1).');
started = tic;
fprintf('[S2-G3] 粗DPD: %d个样本, %d个子带, 网格%d×%d。\n', ...
    N, N_sub, num_grid, num_grid);

for trial = 1:N
    sig_complex = squeeze(double(data.sig_rcv_real_all(trial, :, :))) + ...
        1j * squeeze(double(data.sig_rcv_imag_all(trial, :, :)));
    if ~all(isfinite(sig_complex), 'all')
        error('S2G3:NonfiniteIQ', '样本%d IQ含NaN/Inf。', trial - 1);
    end
    data_fft = fftshift(fft(sig_complex, [], 2), 2);
    data_fft_batch = complex(zeros(N_sub, rcv_num, len));
    for k = 1:N_sub
        for station = 1:rcv_num
            fft_k = data_fft(station, :) .* sub_masks(k, :);
            sig_k = ifft(ifftshift(fft_k));
            power_k = mean(abs(sig_k).^2);
            if power_k > 0
                sig_k = sig_k / sqrt(power_k);
            end
            data_fft_batch(k, station, :) = fftshift(fft(sig_k));
        end
    end
    mtr = DPD_calculator_gpu_batch( ...
        rcvPos, data_fft_batch, init_pos, edge, lamda, fs);
    if ~all(isfinite(mtr), 'all')
        error('S2G3:NonfiniteDPD', '样本%d粗DPD含NaN/Inf。', trial - 1);
    end
    mtr_sub_all(trial, :, :, :) = single(mtr);
    if mod(trial, 16) == 0 || trial == N
        fprintf('[S2-G3] %d/%d, elapsed=%.1fs\n', trial, N, toc(started));
    end
end

src_count_all = data.src_count_all(1:N, :);
band_mask_all = data.band_mask_all(1:N, :, :);
ignore_mask_all = data.ignore_mask_all(1:N, :, :);
avg_snr_all = data.avg_snr_all(1:N, :);
fc_offset_all = data.fc_offset_all(1:N, :);
src_pos_all = data.src_pos_all(1:N, :, :);
symbolRate_all = data.symbolRate_all(1:N, :);
BW_actual_all = data.BW_actual_all(1:N, :);
N_sub_val = int32(N_sub);
max_src_val = int32(max_src);
B_win_val = single(data.B_win_val);
B_step_val = single(data.B_step_val);
fs_val = single(fs);
sub_f_lo_val = single(sub_f_lo);
sub_f_hi_val = single(sub_f_hi);
thresh_val = single(data.thresh_val);
num_count_classes = int32(data.num_count_classes);
num_grid_val = int32(num_grid);
edge_val = single(edge);
lamda_val = single(lamda);
source_iq_path = string(input_mat);
s2g3_algorithm = "third_chapter_exact_19_subband_dpd";
elapsed_seconds = toc(started);

output_parent = fileparts(output_mat);
if ~isempty(output_parent) && ~isfolder(output_parent)
    mkdir(output_parent);
end
save(output_mat, ...
    'mtr_sub_all', 'src_count_all', 'band_mask_all', 'ignore_mask_all', ...
    'avg_snr_all', 'fc_offset_all', 'src_pos_all', ...
    'symbolRate_all', 'BW_actual_all', 'sample_idx_all', ...
    'N_sub_val', 'max_src_val', 'num_count_classes', ...
    'B_win_val', 'B_step_val', 'fs_val', 'sub_f_lo_val', 'sub_f_hi_val', ...
    'thresh_val', 'num_grid_val', 'edge_val', 'lamda_val', ...
    'source_iq_path', 's2g3_algorithm', 'elapsed_seconds', '-v7.3');
fprintf('[S2-G3] 粗DPD已保存: %s (%.1fs)\n', output_mat, elapsed_seconds);
end
