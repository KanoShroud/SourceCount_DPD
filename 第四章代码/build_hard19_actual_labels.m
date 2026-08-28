function [band_mask, ignore_mask] = build_hard19_actual_labels( ...
    fc_offset, bw_actual, n_src, max_src, sub_f_lo, sub_f_hi, b_win, threshold)
%BUILD_HARD19_ACTUAL_LABELS 由实际频段生成Hard-19标签。
%
% 正例：实际频段与10 MHz子带的覆盖率不小于threshold。
% ignore：存在实际频段覆盖，但覆盖率尚未达到threshold。

arguments
    fc_offset (1,:) double
    bw_actual (1,:) double
    n_src (1,1) double {mustBeInteger, mustBeNonnegative}
    max_src (1,1) double {mustBeInteger, mustBePositive}
    sub_f_lo (1,:) double
    sub_f_hi (1,:) double
    b_win (1,1) double {mustBePositive}
    threshold (1,1) double {mustBeGreaterThanOrEqual(threshold, 0), ...
        mustBeLessThanOrEqual(threshold, 1)}
end

if n_src > max_src || numel(fc_offset) < n_src || numel(bw_actual) < n_src
    error('S2G5R1:InvalidSourceShape', '有效信源数量与频带参数shape不一致。');
end
if numel(sub_f_lo) ~= numel(sub_f_hi) || any(sub_f_hi <= sub_f_lo)
    error('S2G5R1:InvalidSubbands', '子带上下边界无效。');
end
if any(bw_actual(1:n_src) <= 0)
    error('S2G5R1:InvalidBandwidth', '有效信源的BW_actual必须大于0。');
end

n_sub = numel(sub_f_lo);
band_mask = zeros(max_src, n_sub, 'single');
ignore_mask = zeros(max_src, n_sub, 'single');
for source_idx = 1:n_src
    actual_lo = fc_offset(source_idx) - bw_actual(source_idx) / 2;
    actual_hi = fc_offset(source_idx) + bw_actual(source_idx) / 2;
    for band_idx = 1:n_sub
        overlap = max(0, min(actual_hi, sub_f_hi(band_idx)) - ...
            max(actual_lo, sub_f_lo(band_idx)));
        coverage = overlap / b_win;
        if coverage >= threshold
            band_mask(source_idx, band_idx) = 1;
        elseif coverage > 0
            ignore_mask(source_idx, band_idx) = 1;
        end
    end
end
end
