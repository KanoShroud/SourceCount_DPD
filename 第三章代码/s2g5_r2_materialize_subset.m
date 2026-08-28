function s2g5_r2_materialize_subset(input_mat, output_mat, manifest_json, subset_name)
%S2G5_R2_MATERIALIZE_SUBSET 按R2 manifest生成CH3兼容MAT子集。

input_mat = char(string(input_mat));
output_mat = char(string(output_mat));
manifest_json = char(string(manifest_json));
subset_name = char(string(subset_name));
if ~isfile(input_mat)
    error('S2G5R2:MissingInput', '输入MAT不存在: %s', input_mat);
end
if isfile(output_mat)
    error('S2G5R2:RefuseOverwrite', '拒绝覆盖已有输出: %s', output_mat);
end
if ~isfile(manifest_json)
    error('S2G5R2:MissingManifest', 'manifest不存在: %s', manifest_json);
end

manifest = jsondecode(fileread(manifest_json));
if ~isfield(manifest, subset_name)
    error('S2G5R2:MissingSubset', 'manifest缺少子集: %s', subset_name);
end
indices_zero_based = double(manifest.(subset_name).indices(:));
if isempty(indices_zero_based) || any(indices_zero_based < 0) || ...
        any(indices_zero_based ~= floor(indices_zero_based)) || ...
        numel(unique(indices_zero_based)) ~= numel(indices_zero_based)
    error('S2G5R2:InvalidIndices', '子集索引为空、越界或重复。');
end
indices = indices_zero_based + 1;

data = load(input_mat);
sample_fields = { ...
    'mtr_sub_all', 'src_count_all', 'band_mask_all', 'ignore_mask_all', ...
    'avg_snr_all', 'fc_offset_all', 'src_pos_all', ...
    'symbolRate_all', 'BW_actual_all', 'sample_idx_all'};
for field_idx = 1:numel(sample_fields)
    field_name = sample_fields{field_idx};
    if ~isfield(data, field_name)
        error('S2G5R2:MissingField', '输入缺少字段: %s', field_name);
    end
    field_value = data.(field_name);
    if max(indices) > size(field_value, 1)
        error('S2G5R2:IndexOutOfRange', '索引超出字段%s的样本维。', field_name);
    end
    subscripts = repmat({':'}, 1, ndims(field_value));
    subscripts{1} = indices;
    data.(field_name) = field_value(subscripts{:});
end
data.subset_name = subset_name;
data.subset_manifest_path = string(manifest_json);
data.parent_coarse_path = string(input_mat);

output_parent = fileparts(output_mat);
if ~isempty(output_parent) && ~isfolder(output_parent)
    mkdir(output_parent);
end
save(output_mat, '-struct', 'data', '-v7.3');
fprintf('[S2-G5-R2] 子集%s已保存: %s (%d条)\n', ...
    subset_name, output_mat, numel(indices));
end
